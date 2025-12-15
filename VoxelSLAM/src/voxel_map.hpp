#ifndef VOXEL_MAP2_HPP
#define VOXEL_MAP2_HPP

#include "tools.hpp"
#include "preintegration.hpp"
#include <thread>
#include <Eigen/Eigenvalues>
#include <unordered_set>
#include <mutex>

#include <ros/ros.h>
#include <fstream>

struct pointVar 
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pnt;
  Eigen::Matrix3d var;
};

using PVec = vector<pointVar>;
using PVecPtr = shared_ptr<vector<pointVar>>;

void down_sampling_pvec(PVec &pvec, double voxel_size, pcl::PointCloud<PointType> &pl_keep)
{
  unordered_map<VOXEL_LOC, pair<pointVar, int>> feat_map;
  float loc_xyz[3];
  for(pointVar &pv: pvec)
  {
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pv.pnt[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter == feat_map.end())
    {
      feat_map[position] = make_pair(pv, 1);
    }
    else
    {
      pair<pointVar, int> &pp = iter->second;
      pp.first.pnt = (pp.first.pnt * pp.second + pv.pnt) / (pp.second + 1);
      pp.first.var = (pp.first.var * pp.second + pv.var) / (pp.second + 1);
      pp.second += 1;
    }
  }

  pcl::PointCloud<PointType>().swap(pl_keep);
  pl_keep.reserve(feat_map.size());
  PointType ap;
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
  {
    pointVar &pv = iter->second.first;
    ap.x = pv.pnt[0]; ap.y = pv.pnt[1]; ap.z = pv.pnt[2];
    ap.normal_x = pv.var(0, 0);
    ap.normal_y = pv.var(1, 1);
    ap.normal_z = pv.var(2, 2);
    pl_keep.push_back(ap);
  }
 
}

struct Plane
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> plane_var;
  float radius = 0;
  bool is_plane = false;

  Plane()
  {
    plane_var.setZero();
  }

};

Eigen::Vector4d min_point;
double min_eigen_value;
int max_layer = 2;
int max_points = 100;
double voxel_size = 1.0;
int min_ba_point = 20;
vector<double> plane_eigen_value_thre;

/**
 * @brief 计算单个点在点簇中的协方差
 * 
 * @param pv 
 * @param bcov 
 * @param vec 
 */
void Bf_var(const pointVar &pv, Eigen::Matrix<double, 9, 9> &bcov, const Eigen::Vector3d &vec)
{
  // 和论文中分成两部分计算，下面还有个 3x3 的单位阵
  Eigen::Matrix<double, 6, 3> Bi;
  // Eigen::Vector3d &vec = pv.world;
  Bi << 2*vec(0),        0,        0,
          vec(1),   vec(0),        0,
          vec(2),        0,   vec(0),
               0, 2*vec(1),        0,
               0,   vec(2),   vec(1),
               0,        0, 2*vec(2);
  Eigen::Matrix<double, 6, 3> Biup = Bi * pv.var;
  bcov.block<6, 6>(0, 0) = Biup * Bi.transpose();
  bcov.block<6, 3>(0, 6) = Biup;
  bcov.block<3, 6>(6, 0) = Biup.transpose();
  bcov.block<3, 3>(6, 6) = pv.var;
}

// The LiDAR BA factor in optimization
class LidarFactor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<PointCluster> sig_vecs;              // 每个 voxel 的固定点簇（历史边缘化累积）
  vector<vector<PointCluster>> plvec_voxels;  // 每个 voxel 在滑窗内各帧的点簇（局部系），形状：[voxel_num][win_size]
  vector<double> coeffs;                      // 每个 voxel 的权重系数（当前实现基本恒为 1）
  PLV(3) eig_values;                          // 每个 voxel 的协方差特征值（最小特征值对应平面厚度）
  PLM(3) eig_vectors;                         // 每个 voxel 的协方差特征向量（col(0) 通常作为平面法向）
  vector<PointCluster> pcr_adds;              // 每个 voxel 的世界系累积点簇（固定点 + 滑窗点变换后）
  int win_size;                               // 滑窗大小

  LidarFactor(int _w): win_size(_w){}

  /**
   * @brief 将一个 voxel 的点簇及其平面特征推入 LidarFactor 缓存
   * 
   * @param vec_orig  滑窗内每帧在该 voxel 的点簇（局部系）
   * @param fix       固定点簇
   * @param coe       权重系数（当前实现恒 1）
   * @param eig_value 平面特征值（法向对应最小特征值）
   * @param eig_vector 平面特征向量（列向量为主轴，col(0) 为法向）
   * @param pcr_add   该 voxel 的累积点簇（世界系，用于协方差/中心）
   */
  void push_voxel(vector<PointCluster> &vec_orig, PointCluster &fix, double coe, Eigen::Vector3d &eig_value, Eigen::Matrix3d &eig_vector, PointCluster &pcr_add)
  {
    plvec_voxels.push_back(vec_orig);
    sig_vecs.push_back(fix);
    coeffs.push_back(coe);
    eig_values.push_back(eig_value);
    eig_vectors.push_back(eig_vector);
    pcr_adds.push_back(pcr_add);
  }

  /**
   * @brief 右扰动更新，对应补充材料中的公式
   * 
   * @param xs 
   * @param head 
   * @param end 
   * @param Hess 
   * @param JacT 
   * @param residual 
   */
  void acc_evaluate2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<PointCluster> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    // 累加不同特征的雅克比和 Hessian
    for(int a=head; a<end; a++)
    {
      vector<PointCluster> &sig_orig = plvec_voxels[a];
      double coe = coeffs[a];

      // PointCluster sig = sig_vecs[a];
      // for(int i=0; i<win_size; i++)
      // if(sig_orig[i].N != 0)
      // {
      //   sig_tran[i].transform(sig_orig[i], xs[i]);
      //   sig += sig_tran[i];
      // }
      
      // const Eigen::Vector3d &vBar = sig.v / sig.N;
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      // const Eigen::Vector3d &lmbd = saes.eigenvalues();
      // const Eigen::Matrix3d &U = saes.eigenvectors();
      // int NN = sig.N;

      Eigen::Vector3d lmbd = eig_values[a];
      Eigen::Matrix3d U = eig_vectors[a];
      int NN = pcr_adds[a].N;
      Eigen::Vector3d vBar = pcr_adds[a].v / NN;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};
      Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i=0; i<3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i=0; i<win_size; i++)
      // for(int i=1; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        Eigen::Matrix3d Pi = sig_orig[i].P;
        Eigen::Vector3d vi = sig_orig[i].v;
        Eigen::Matrix3d Ri = xs[i].R;
        double ni = sig_orig[i].N;

        Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
        Eigen::Vector3d RiTuk = Ri.transpose() * uk;
        Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

        Eigen::Vector3d PiRiTuk = Pi * RiTuk;
        viRiTuk[i] = vihat * RiTuk;
        viRiTukukT[i] = viRiTuk[i] * uk.transpose();
        
        Eigen::Vector3d ti_v = xs[i].p - vBar;
        double ukTti_v = uk.dot(ti_v);

        Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
        Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
        Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
        Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
        Auk[i] /= NN;

        const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
        JacT.block<6, 1>(6*i, 0) += coe * jjt;

        // 计算对角部分 Hessian
        const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
        Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
        Hb.block<3, 3>(0, 0) += 2.0/NN * (combo1 - RiTukhat*Pi) * RiTukhat - 2.0/NN/NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

        Hess.block<6, 6>(6*i, 6*i) += coe * Hb;
      }

      // 计算非对角部分 Hessian
      for(int i=0; i<win_size-1; i++)
      // for(int i=1; i<win_size-1; i++)
      if(sig_orig[i].N != 0)
      {
        double ni = sig_orig[i].N;
        for(int j=i+1; j<win_size; j++)
        if(sig_orig[j].N != 0)
        {
          double nj = sig_orig[j].N;
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
          Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
          Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
          Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
          Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

          Hess.block<6, 6>(6*i, 6*j) += coe * Hb;
        }
      }
      
      residual += coe * lmbd[kk];
    }

    // 补全 Hessian
    for(int i=1; i<win_size; i++)
      for(int j=0; j<i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  /**
   * @brief 仅计算 LiDAR 部分残差（不求导），并用当前状态更新每个 voxel 的统计量
   *
   * 说明：
   * - 本函数会把每个 voxel 在滑窗内各帧的点簇（局部系）按当前位姿 xs[i] 变换到世界系，
   *   得到累积点簇 sig，并对其协方差做特征分解。
   * - 同时会把计算得到的 eig_values/eig_vectors/pcr_adds 写回缓存，供后续边缘化阶段复用。
   *
   * @param xs       当前滑窗状态
   * @param head     处理的 voxel 起始索引（多线程分块）
   * @param end      处理的 voxel 结束索引（不含）
   * @param residual 输出：该分块内的残差累加
   */
  void evaluate_only_residual(const vector<IMUST> &xs, int head, int end, double &residual)
  {
    residual = 0;
    // vector<PointCluster> sig_tran(win_size);
    int kk = 0; // 取第 kk 个特征值作为残差项（0 对应最小特征值，近似平面厚度）

    // int gps_size = plvec_voxels.size();
    PointCluster pcr; // 临时点簇：用于把某帧点簇从局部系变换到世界系

    for(int a=head; a<end; a++)
    {
      // sig_orig：该 voxel 在滑窗内每一帧的点簇统计（局部系）
      const vector<PointCluster> &sig_orig = plvec_voxels[a];
      // sig：从固定点簇开始累积（历史边缘化帧贡献）
      //! 这里的sig是临时变量，不会影响到sig_vecs[a]，也不会影响到下一次迭代
      PointCluster sig = sig_vecs[a];

      for(int i=0; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        // 将第 i 帧的点簇从局部系变换到世界系，并累加进总点簇 sig
        pcr.transform(sig_orig[i], xs[i]);
        sig += pcr;
      }

      // 计算点簇均值与协方差：cov = E[xx^T] - mu mu^T
      Eigen::Vector3d vBar = sig.v / sig.N;
      // Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P - sig.v * vBar.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();

      // 将本次计算得到的平面统计量写回缓存：
      // - eig_values/eig_vectors：后续求导/边缘化会复用
      // - pcr_adds：该 voxel 的“世界系累积点簇”（窗口点 + 固定点）
      eig_values[a] = saes.eigenvalues();
      eig_vectors[a] = saes.eigenvectors();
      pcr_adds[a] = sig;
      // Ns[a] = sig.N;

      // 残差：以最小特征值衡量点簇“平面厚度”（越小越接近平面）
      residual += coeffs[a] * lmbd[kk];
    }
    
  }

  void clear()
  {
    sig_vecs.clear(); plvec_voxels.clear();
    eig_values.clear(); eig_vectors.clear();
    pcr_adds.clear(); coeffs.clear();
  }

  ~LidarFactor(){}

};

// The LM optimizer for LiDAR BA
class Lidar_BA_Optimizer
{
public:
  int win_size, jac_leng, thd_num = 2;

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    // int thd_num = 4;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num); 
    PLV(-1) jacobins(thd_num);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    // for(int i=0; i<tthd_num; i++)
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for(int i=0; i<tthd_num; i++)
    {
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    double residual1 = 0;
    // voxhess.evaluate_only_residual(x_stats, 0, voxhess.plvec_voxels.size(), residual1);

    // int thd_num = 2;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      printf("Too Less Voxel"); exit(0);
    }
    vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
        mthreads[i]->join();
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual1 += residuals[i];
      delete mthreads[i];
    }

    return residual1;
  }

  bool damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd* hess, vector<double> &resis, int max_iter = 3, bool is_display = false)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);
    hess->resize(jac_leng, jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    bool is_converge = true;

    // double tt1 = ros::Time::now().toSec();
    // for(int i=0; i<10; i++)
    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        residual1 = divide_thread(x_stats, voxhess, Hess, JacT);
        *hess = Hess;
      }

      if(i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();
      
      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(6*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(6*j+3, 0);
      }
      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);

      residual2 = only_residual(x_stats_temp, voxhess);

      q = (residual1-residual2);
      if(is_display)
        printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
        is_converge = false;
      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }
    resis.push_back(residual2);
    return is_converge;
  }

};

double imu_coef = 1e-4;
// double imu_coef = 1e-8;
#define DVEL 6
// The LiDAR-Inertial BA optimizer
class LI_BA_Optimizer
{
public:
  int win_size, jac_leng, imu_leng;

  /**
   * @brief 累加 IMU 和不同线程中平面特征的雅克比和 Hessian 矩阵
   * 
   * @param Hess 
   * @param JacT 
   * @param hs 
   * @param js 
   */
  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for(int i=0; i<win_size; i++)
    {
      JacT.block<DVEL, 1>(i*DIM, 0) += js.block<DVEL, 1>(i*DVEL, 0);
      for(int j=0; j<win_size; j++)
        Hess.block<DVEL, DVEL>(i*DIM, j*DIM) += hs.block<DVEL, DVEL>(i*DVEL, j*DVEL);
    }
  }

  /**
   * @brief 多线程累加 LiDAR+IMU 的雅克比/Hessian，返回总残差
   * 
   * @param x_stats     滑窗状态
   * @param voxhess     LidarFactor（平面因子集合）
   * @param imus_factor IMU 预积分因子
   * @param Hess        输出：联合 Hessian
   * @param JacT        输出：联合雅可比向量
   * @return double     残差和（IMU*coef + LiDAR）
   */
  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);  // 各线程局部 Hessian
    PLV(-1) jacobins(thd_num);  // 各线程局部 JacT
    vector<double> resis(thd_num, 0); // 各线程残差

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    vector<thread*> mthreads(tthd_num);
    // for(int i=0; i<tthd_num; i++)
    // 计算BALM的雅克比矩阵和Hessian矩阵
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part * (i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    // IMU 部分：预积分雅可比/Hessian 累加
    for(int i=0; i<win_size-1; i++)
    {
      jtj.setZero(); gg.setZero();
      residual += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, true);
      Hess.block<DIM*2, DIM*2>(i*DIM, i*DIM) += jtj;
      JacT.block<DIM*2, 1>(i*DIM, 0) += gg;
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity(); rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual *= (imu_coef * 0.5);

    // printf("resi: %lf\n", residual);

    // LiDAR 部分：等待多线程结束，累加各线程结果
    for(int i=0; i<tthd_num; i++)
    {
      // mthreads[i]->join();
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i];
      delete mthreads[i];
    }

    // 返回残差
    return residual;
  }

  /**
   * @brief 在给定状态下计算总残差（IMU + LiDAR），但不计算导数
   * 
   * 说明：
   * - IMU 部分通过预积分因子 give_evaluate(..., false) 仅累计残差；
   * - LiDAR 部分通过 LidarFactor::evaluate_only_residual 仅累计残差，
   *   且会把该状态下的平面统计量（eig_values/eig_vectors/pcr_adds）写回 voxhess 缓存，更新平面参数，
   *   供后续边缘化阶段复用。
   *
   * @param x_stats     滑窗状态
   * @param voxhess     LidarFactor（平面因子集合）
   * @param imus_factor IMU 预积分因子
   * @return double     总残差（IMU*coef + LiDAR）
   */
  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor)
  {
    double residual1 = 0, residual2 = 0;
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    int thd_num = 5;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      // printf("Too Less Voxel"); exit(0);
      thd_num = 1;
    }
    vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    // 多线程计算 LiDAR 残差（每个线程处理一段 voxel）
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    // IMU 残差：相邻帧预积分约束（不求导）
    for(int i=0; i<win_size-1; i++)
      residual1 += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    // 汇总 LiDAR 残差并等待线程结束；第 0 段在主线程计算
    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual2 += residuals[i];
    }

    return (residual1 + residual2);
  }

  /**
   * @brief LiDAR-IMU 联合滑窗优化（LM 法），更新位姿/速度/bias
   * 
   * @param x_stats      滑窗状态向量（会被原地更新）
   * @param voxhess      LidarFactor（平面因子集合）
   * @param imus_factor  IMU 预积分因子
   * @param hess         输出：最新迭代的 Hessian（可用于信息矩阵）
   */
  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd* hess)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;                  // 只优化 R/p（每帧6维）时的长度
    imu_leng = win_size * DIM;                // 全状态长度（R p v bg ba，DIM=15）
    double u = 0.01, v = 2;                   // LM 阻尼系数及放大因子
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng); // LM 对角/雅可比累加矩阵
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);                   // 雅可比向量和状态增量
    hess->resize(imu_leng, imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    double hesstime = 0;
    double resitime = 0;
  
    // LM 迭代（默认 3 次，可调）
    // for(int i=0; i<10; i++)
    for(int i=0; i<3; i++)
    {
      if(is_calc_hess)
      {
        // 计算 Jacobian/Hessian 与当前残差
        double tm = ros::Time::now().toSec();
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        hesstime += ros::Time::now().toSec() - tm;
        *hess = Hess;
      }
      
      // 固定住滑窗的第一帧
      Hess.topRows(DIM).setZero();
      Hess.leftCols(DIM).setZero();
      Hess.block<DIM, DIM>(0, 0).setIdentity();
      JacT.head(DIM).setZero();

      // LM法求解，马夸尔特方法，椭球形信赖域
      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      // 更新滑窗状态（右扰动）
      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM*j+3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM*j+6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM*j+9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM*j+12, 0);
      }

      for(int j=0; j<win_size-1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM*j, 0));

      double q1 = 0.5 * dxi.dot(u*D*dxi-JacT);

      double tl1 = ros::Time::now().toSec();
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      double tl2 = ros::Time::now().toSec();
      // printf("onlyresi: %lf\n", tl2-tl1);
      resitime += tl2 - tl1;

      q = (residual1-residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      // Nielsen 法调整阻尼系数 u/v，接受或回退
      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for(int j=0; j<win_size-1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }

      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }

    // printf("ba: %lf %lf %zu\n", hesstime, resitime, voxhess.plvec_voxels.size());

  }

};

// The LiDAR-Inertial BA optimizer with gravity optimization
class LI_BA_OptimizerGravity
{
public:
  int win_size, jac_leng, imu_leng;

  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for(int i=0; i<win_size; i++)
    {
      JacT.block<DVEL, 1>(i*DIM, 0) += js.block<DVEL, 1>(i*DVEL, 0);
      for(int j=0; j<win_size; j++)
        Hess.block<DVEL, DVEL>(i*DIM, j*DIM) += hs.block<DVEL, DVEL>(i*DVEL, j*DVEL);
    }
  }

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num); 
    PLV(-1) jacobins(thd_num);
    vector<double> resis(thd_num, 0);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size(); // 平面因子数量
    if(g_size < tthd_num) tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    vector<thread*> mthreads(tthd_num);
    // for(int i=0; i<tthd_num; i++)
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part * (i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    Eigen::MatrixXd jtj(2*DIM+3, 2*DIM+3);
    Eigen::VectorXd gg(2*DIM+3);

    for(int i=0; i<win_size-1; i++)
    {
      jtj.setZero(); gg.setZero();
      residual += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i+1], jtj, gg, true);
      Hess.block<DIM*2, DIM*2>(i*DIM, i*DIM) += jtj.block<2*DIM, 2*DIM>(0, 0);
      Hess.block<DIM*2, 3>(i*DIM, imu_leng-3) += jtj.block<2*DIM, 3>(0, 2*DIM);
      Hess.block<3, DIM*2>(imu_leng-3, i*DIM) += jtj.block<3, 2*DIM>(2*DIM,0);
      Hess.block<3, 3>(imu_leng-3, imu_leng-3) += jtj.block<3, 3>(2*DIM, 2*DIM);

      JacT.block<DIM*2, 1>(i*DIM, 0) += gg.head(2*DIM);
      JacT.tail(3) += gg.tail(3);
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity(); rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual *= (imu_coef * 0.5);

    // printf("resi: %lf\n", residual);

    for(int i=0; i<tthd_num; i++)
    {
      // mthreads[i]->join();
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor)
  {
    double residual1 = 0, residual2 = 0;
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    int thd_num = 5;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      // printf("Too Less Voxel"); exit(0);
      thd_num = 1;
    }
    vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<win_size-1; i++)
      residual1 += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i+1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual2 += residuals[i];
    }

    return (residual1 + residual2);
  }

  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, vector<double> &resis, Eigen::MatrixXd* hess, int max_iter = 2)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;
    imu_leng = win_size * DIM + 3;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng);
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;
    
    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        *hess = Hess;
      }

      if(i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();

      // Hess.rightCols(3).setZero();
      // Hess.bottomRows(3).setZero();
      // Hess.block<3, 3>(imu_leng-3, imu_leng-3).setIdentity();
      // JacT.tail(3).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      x_stats_temp[0].g += dxi.tail(3);
      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM*j+3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM*j+6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM*j+9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM*j+12, 0);
        x_stats_temp[j].g = x_stats_temp[0].g;
      }

      for(int j=0; j<win_size-1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM*j, 0));
      
      double q1 = 0.5 * dxi.dot(u*D*dxi-JacT);
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      q = (residual1-residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);
      
      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for(int j=0; j<win_size-1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }
      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;

    }
    resis.push_back(residual2);
    
  }

};

// 10 scans merge into a keyframe
struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IMUST x0;
  pcl::PointCloud<PointType>::Ptr plptr;
  int exist;
  int id, mp;
  float jour;

  Keyframe(IMUST &_x0): x0(_x0), exist(0)
  {
    plptr.reset(new pcl::PointCloud<PointType>());
  }

  void generate(pcl::PointCloud<PointType> &pl_send, Eigen::Matrix3d rot = Eigen::Matrix3d::Identity(), Eigen::Vector3d tra = Eigen::Vector3d(0, 0, 0))
  {
    Eigen::Vector3d v3;
    for(PointType ap: plptr->points)
    {
      v3 << ap.x, ap.y, ap.z;
      v3 = rot * v3 + tra;
      ap.x = v3[0]; ap.y = v3[1]; ap.z = v3[2];
      pl_send.push_back(ap);
    }
  }

};

// The sldingwindow in each voxel nodes
class SlideWindow
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<PVec> points;                  // 滑动窗口中每一帧的点云数据
  vector<PointCluster> pcrs_local;      // 滑动窗口中每帧点云的点簇信息

  SlideWindow(int wdsize)
  {
    pcrs_local.resize(wdsize);
    points.resize(wdsize);
    for(int i=0; i<wdsize; i++)
      points[i].reserve(20);
  }

  void resize(int wdsize)
  {
    if(points.size() != wdsize)
    {
      points.resize(wdsize);
      pcrs_local.resize(wdsize);
    }
  }

  void clear()
  {
    int wdsize = points.size();
    for(int i=0; i<wdsize; i++)
    {
      points[i].clear();
      pcrs_local[i].clear();
    }
  }

};

// The octotree map for odometry and local mapping
// You can re-write it in your own project
// mp 是一个全局的映射表，用来把“滑窗的逻辑帧序号 ord”映射到 SlideWindow 内部数组的存储槽位
int* mp;
class OctoTree
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SlideWindow* sw = nullptr;
  PointCluster pcr_add; // 滑窗中所有点在世界系下的点簇信息 + 固定点的点簇信息
  Eigen::Matrix<double, 9, 9> cov_add;

  PointCluster pcr_fix; // 固定点的点簇信息，被边缘化后退出滑窗的点
  PVec point_fix;

  int layer;
  int octo_state;   // octo_state 0 is end of tree, 1 is not
  int wdsize;   // 滑窗大小
  OctoTree* leaves[8];
  double voxel_center[3];
  double jour = 0;
  float quater_length;

  Plane plane;
  bool isexist = false;

  Eigen::Vector3d eig_value;    // 平面特征值
  Eigen::Matrix3d eig_vector;   // 平面特征向量

  int last_num = 0, opt_state = -1; // voxel序号
  mutex mVox;

  OctoTree(int _l, int _w) : layer(_l), wdsize(_w), octo_state(0)
  {
    for(int i=0; i<8; i++) leaves[i] = nullptr;
    cov_add.setZero();

    // ins = 255.0*rand()/(RAND_MAX + 1.0f);
  }

  /**
   * @brief 在叶子节点中插入一个点，并更新点簇/协方差与滑窗缓存
   * 
   * @param ord  当前点所属的滑窗帧序号
   * @param pv   点（IMU系，含协方差）
   * @param pw   点的世界坐标
   * @param sws  当前线程的 SlideWindow 池（复用避免频繁 new）
   */
  inline void push(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow*> &sws)
  {
    mVox.lock();
    if(sw == nullptr)
    {
      // 池里还有就取一个
      if(sws.size() != 0)
      {
        sw = sws.back();
        sws.pop_back();
        sw->resize(wdsize);
      }
      // 池子空了就新建一个
      else
        sw = new SlideWindow(wdsize);
    }
    if(!isexist) isexist = true;

    // 滑窗的第几帧数据
    int mord = mp[ord];
    if(layer < max_layer)
      sw->points[mord].push_back(pv);
    sw->pcrs_local[mord].push(pv.pnt);
    pcr_add.push(pw);                // 点簇累加（世界系）
    // 累加点簇协方差，用于后续平面拟合
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pw);
    cov_add += Bi;
    mVox.unlock();
  }

  /**
   * @brief 添加固定点，更新点簇协方差
   * 
   * @param pv 
   */
  inline void push_fix(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pv.pnt);
    cov_add += Bi;
  }

  inline void push_fix_novar(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
  }

  inline bool plane_judge(Eigen::Vector3d &eig_values)
  {
    // return (eig_values[0] < min_eigen_value);
    return (eig_values[0] < min_eigen_value && (eig_values[0]/eig_values[2])<plane_eigen_value_thre[layer]);
  }

  /**
   * @brief 递归往 OctoTree 中插入一个点：若当前为叶子则 push，否则下分八叉树
   * 
   * @param ord 当前点所属的滑窗帧序号
   * @param pv  点（IMU系，含协方差）
   * @param pw  点的世界坐标
   * @param sws 某个线程中的滑窗池
   */
  void allocate(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow*> &sws)
  {
    if(octo_state == 0)
    {
      push(ord, pv, pw, sws);
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize); 
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate(ord, pv, pw, sws);
    }

  }

  void allocate_fix(pointVar &pv)
  {
    if(octo_state == 0)
    {
      push_fix_novar(pv);
    }
    else if(layer < max_layer)
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate_fix(pv);
    }
  }

  /**
   * @brief 把固定点云数据分配到各个子节点
   * 
   * @param sws 
   */
  void fix_divide(vector<SlideWindow*> &sws)
  {
    for(pointVar &pv: point_fix)
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->push_fix(pv);
    }

  }

  /**
   * @brief 将当前节点某一帧的点根据世界坐标下所在象限划分到 8 个子节点
   * 
   * @param si  滑窗中的帧序号
   * @param xx  对应帧的位姿，用于把点从 IMU 系转到世界系
   * @param sws 当前线程的滑窗池，子节点 push 时复用
   */
  void subdivide(int si, IMUST &xx, vector<SlideWindow*> &sws)
  {
    for(pointVar &pv: sw->points[mp[si]])
    {
      Eigen::Vector3d pw = xx.R * pv.pnt + xx.p;
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->push(si, pv, pw, sws);
    }
  }

  void plane_update()
  {
    plane.center = pcr_add.v / pcr_add.N;
    int l = 0;
    Eigen::Vector3d u[3] = {eig_vector.col(0), eig_vector.col(1), eig_vector.col(2)};
    double nv = 1.0 / pcr_add.N;

    Eigen::Matrix<double, 3, 9> u_c; u_c.setZero();
    for(int k=0; k<3; k++)
    if(k != l)
    {
      Eigen::Matrix3d ukl = u[k] * u[l].transpose();
      Eigen::Matrix<double, 1, 9> fkl;
      fkl.head(6) << ukl(0, 0), ukl(1, 0)+ukl(0, 1), ukl(2, 0)+ukl(0, 2), 
                     ukl(1, 1), ukl(1, 2)+ukl(2, 1),           ukl(2, 2);
      fkl.tail(3) = -(u[k].dot(plane.center) * u[l] + u[l].dot(plane.center) * u[k]);
      
      u_c += nv / (eig_value[l]-eig_value[k]) * u[k] * fkl;
    }

    Eigen::Matrix<double, 3, 9> Jc = u_c * cov_add;
    plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
    Eigen::Matrix3d Jc_N = nv * Jc.block<3, 3>(0, 6);
    plane.plane_var.block<3, 3>(0, 3) = Jc_N;
    plane.plane_var.block<3, 3>(3, 0) = Jc_N.transpose();
    plane.plane_var.block<3, 3>(3, 3) = nv * nv * cov_add.block<3, 3>(6, 6);
    plane.normal = u[0];
    plane.radius = eig_value[2];
  }

  /**
   * @brief 递归拟合平面：叶子节点尝试拟合，否则分裂下沉到子节点
   * 
   * @param win_count 当前滑窗帧数
   * @param x_buf     滑窗状态，用于分裂时将点转到世界系
   * @param sws       当前线程的滑窗池（分裂后归还复用）
   */
  void recut(int win_count, vector<IMUST> &x_buf, vector<SlideWindow*> &sws)
  {
    if(octo_state == 0)
    {
      if(layer >= 0)
      {
        opt_state = -1;
        // 数量不足直接判定不可拟合
        if(pcr_add.N <= min_point[layer])
        {
          plane.is_plane = false; return;
        }
        if(!isexist || sw == nullptr) return;

        // 拟合平面
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
        eig_value  = saes.eigenvalues();
        eig_vector = saes.eigenvectors();
        plane.is_plane = plane_judge(eig_value);

        // 拟合成功则保留为叶子，结束递归
        if(plane.is_plane)
        {
          return;
        }
        else if(layer >= max_layer)
          return;
      }
      
      if(pcr_fix.N != 0)
      {
        fix_divide(sws);
        // point_fix.clear();
        PVec().swap(point_fix);
      }

      // 拟合失败：按 8 分裂，把当前节点的各帧点分配到子节点
      for(int i=0; i<win_count; i++)
        subdivide(i, x_buf[i], sws);

      // 分裂后归还滑窗缓存，标记为内部节点
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
      octo_state = 1;
    }

    // 递归对子节点继续尝试平面拟合
    for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(win_count, x_buf, sws);

  }

  /**
   * @brief 递归边缘化：移除滑窗最旧帧的贡献，将其转为固定点并更新平面信息
   * 
   * @param win_count 当前滑窗帧数
   * @param mgsize    边缘化帧数（通常 1）
   * @param x_buf     滑窗状态
   * @param vox_opt   LidarFactor（保存前一次优化的平面信息，可复用）
   */
  void margi(int win_count, int mgsize, vector<IMUST> &x_buf, const LidarFactor &vox_opt)
  {
    if(octo_state == 0 && layer>=0)
    {
      if(!isexist || sw == nullptr) return;
      mVox.lock();
      vector<PointCluster> pcrs_world(wdsize);
      // pcr_add = pcr_fix;
      // for(int i=0; i<win_count; i++)
      // if(sw->pcrs_local[mp[i]].N != 0)
      // {
      //   pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
      //   pcr_add += pcrs_world[i];
      // }

      if(opt_state >= int(vox_opt.pcr_adds.size()))
      {
        printf("Error: opt_state: %d %zu\n", opt_state, vox_opt.pcr_adds.size());
        exit(0);
      }

      //! 在 only_residual 函数中会更新优化后的平面信息，所以这里可以直接复用
      if(opt_state >= 0)
      {
        // 如果优化时已经提取过因子，直接复用当时的点簇/特征
        pcr_add = vox_opt.pcr_adds[opt_state];
        eig_value  = vox_opt.eig_values[opt_state];
        eig_vector = vox_opt.eig_vectors[opt_state];
        opt_state = -1;
        
        for(int i=0; i<mgsize; i++)
        if(sw->pcrs_local[mp[i]].N != 0)
        {
          pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
        }
      }
      else
      {
        // 否则根据当前滑窗重新累计点簇并拟合平面
        pcr_add = pcr_fix;
        for(int i=0; i<win_count; i++)
        if(sw->pcrs_local[mp[i]].N != 0)
        {
          pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
          pcr_add += pcrs_world[i];
        }

        if(plane.is_plane)
        {
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
          eig_value = saes.eigenvalues();
          eig_vector = saes.eigenvectors();
        }
        
      }

      if(pcr_fix.N < max_points && plane.is_plane)
      if(pcr_add.N - last_num >= 5 || last_num <= 10)
      {
        plane_update();
        last_num = pcr_add.N;
      }

      // 将被边缘化的帧转换为固定点簇，累加到 pcr_fix/point_fix
      if(pcr_fix.N < max_points)
      {
        for(int i=0; i<mgsize; i++)
        if(pcrs_world[i].N != 0)
        {
          pcr_fix += pcrs_world[i];
          for(pointVar pv: sw->points[mp[i]])
          {
            pv.pnt = x_buf[i].R * pv.pnt + x_buf[i].p;
            point_fix.push_back(pv);
          }
        }

      }
      else
      {
        // 固定点过多时只移除边缘化帧贡献，不再累加固定点
        for(int i=0; i<mgsize; i++)
          if(pcrs_world[i].N != 0)
            pcr_add -= pcrs_world[i];
        
        if(point_fix.size() != 0)
          PVec().swap(point_fix);
      }

      for(int i=0; i<mgsize; i++)
      if(sw->pcrs_local[mp[i]].N != 0)
      {
        sw->pcrs_local[mp[i]].clear();
        sw->points[mp[i]].clear();
      }
      
      // 如果剩下的点数小于固定点数，就删除该voxel
      if(pcr_fix.N >= pcr_add.N)
        isexist = false;
      else
        isexist = true;
      
      mVox.unlock();
    }
    else
    {
      isexist = false;
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->margi(win_count, mgsize, x_buf, vox_opt);
        isexist = isexist || leaves[i]->isexist;
      }
    }

  }

  // Extract the LiDAR factor
  /**
   * @brief 递归提取可用的平面点簇，生成 LidarFactor 因子
   * 
   * @param vox_opt LidarFactor 容器，收集每个 voxel 的点簇/特征
   */
  void tras_opt(LidarFactor &vox_opt)
  {
    if(octo_state == 0)
    {
      if(layer >= 0 && isexist && plane.is_plane && sw!=nullptr)
      {
        // 平面比较厚，放弃
        if(eig_value[0]/eig_value[1] > 0.12) return;

        double coe = 1;
        vector<PointCluster> pcrs(wdsize);
        for(int i=0; i<wdsize; i++)
          pcrs[i] = sw->pcrs_local[mp[i]];
        opt_state = vox_opt.plvec_voxels.size();
        // 把滑窗中每一帧的点簇信息都存起来
        vox_opt.push_voxel(pcrs, pcr_fix, coe, eig_value, eig_vector, pcr_add);
      }

    }
    else
    {
      // 当前不是叶子节点就递归搜索
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
    }


  }

  /**
   * @brief 在对应的voxel中计算点是否有匹配的平面
   * 
   * @param wld 点的世界坐标
   * @param pla 
   * @param max_prob 
   * @param var_wld 
   * @param sigma_d 
   * @param oc 
   * @return int 
   */
  int match(Eigen::Vector3d &wld, Plane* &pla, double &max_prob, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc)
  {
    int flag = 0;
    // 已经到达最大层数
    if(octo_state == 0)
    {
      if(plane.is_plane)
      {
        float dis_to_plane = fabs(plane.normal.dot(wld - plane.center));
        float dis_to_center = (plane.center - wld).squaredNorm();
        float range_dis = (dis_to_center - dis_to_plane * dis_to_plane);
        // 在平面上的投影点距离平面中心小于 3σ
        if(range_dis <= 3*3*plane.radius)
        {
          // 计算点面距离协方差
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = wld - plane.center;
          J_nq.block<1, 3>(0, 3) = -plane.normal;
          double sigma_l = J_nq * plane.plane_var * J_nq.transpose();
          sigma_l += plane.normal.transpose() * var_wld * plane.normal;
          // 点面距离协方差小于 3σ
          if(dis_to_plane < 3 * sqrt(sigma_l))
          {
            // float prob = 1 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
            // if(prob > max_prob)
            {
              oc = this;
              sigma_d = sigma_l;
              // max_prob = prob;
              pla = &plane;
            }

            flag = 1;
          }
        }
      }
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(wld[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      // for(int i=0; i<8; i++)
      // if(leaves[i] != nullptr)
      // {
      //   int flg = leaves[i]->match(wld, pla, max_prob, var_wld);
      //   if(i == leafnum)
      //     flag = flg;
      // }

      if(leaves[leafnum] != nullptr)
        flag = leaves[leafnum]->match(wld, pla, max_prob, var_wld, sigma_d, oc);

      // for(int i=0; i<8; i++)
      //   if(leaves[i] != nullptr)
      //     leaves[i]->match(pv, pla, max_prob, var_wld);
    }

    return flag;
  }

  void tras_ptr(vector<OctoTree*> &octos_release)
  {
    if(octo_state == 1)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        octos_release.push_back(leaves[i]);
        leaves[i]->tras_ptr(octos_release);
      }
    }
  }

  // ~OctoTree()
  // {
  //   for(int i=0; i<8; i++)
  //   if(leaves[i] != nullptr)
  //   {
  //     delete leaves[i];
  //     leaves[i] = nullptr;
  //   }
  // }

  // Extract the point cloud map for debug
  void tras_display(int win_count, pcl::PointCloud<PointType> &pl_fixd, pcl::PointCloud<PointType> &pl_wind, vector<IMUST> &x_buf)
  {
    if(octo_state == 0)
    {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
      Eigen::Matrix3d eig_vectors = saes.eigenvectors();
      Eigen::Vector3d eig_values  = saes.eigenvalues();

      PointType ap; 
      // ap.intensity = ins;

      if(plane.is_plane)
      {
        // if(pcr_add.N-pcr_fix.N < min_ba_point) return;
        // if(eig_value[0]/eig_value[1] > 0.1)
        //   return;

        // for(pointVar &pv: point_fix)
        // {
        //   Eigen::Vector3d pvec = pv.pnt;
        //   ap.x = pvec[0];
        //   ap.y = pvec[1];
        //   ap.z = pvec[2];
        //   ap.normal_x = sqrt(eig_values[0]);
        //   ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
        //   ap.normal_z = pcr_add.N;
        //   ap.curvature = pcr_add.N - pcr_fix.N;
        //   pl_fixd.push_back(ap);
        // }

        for(int i=0; i<win_count; i++)
        for(pointVar &pv: sw->points[mp[i]])
        {
          Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
          ap.x = pvec[0]; ap.y = pvec[1]; ap.z = pvec[2];
          // ap.normal_x = sqrt(eig_values[0]);
          // ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
          // ap.normal_z = pcr_add.N;
          // ap.curvature = pcr_add.N - pcr_fix.N;
          pl_wind.push_back(ap);
        }
      }

    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(win_count, pl_fixd, pl_wind, x_buf);
    }

  }

  bool inside(Eigen::Vector3d &wld)
  {
    double hl = quater_length * 2;
    return (wld[0] >= voxel_center[0] - hl &&
            wld[0] <= voxel_center[0] + hl &&
            wld[1] >= voxel_center[1] - hl &&
            wld[1] <= voxel_center[1] + hl &&
            wld[2] >= voxel_center[2] - hl &&
            wld[2] <= voxel_center[2] + hl);
  }

  void clear_slwd(vector<SlideWindow*> &sws)
  {
    if(octo_state != 0)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->clear_slwd(sws);
      }
    }

    if(sw != nullptr)
    {
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
    }

  }

};

/**
 * @brief 单线程将当前帧点云按世界坐标划分到 voxel，并更新对应 OctoTree/滑窗
 * 
 * @param feat_map      全局体素地图（voxel->OctoTree）
 * @param pvec          当前帧点云（IMU系，带协方差）
 * @param win_count     当前帧在滑窗中的逻辑序号
 * @param feat_tem_map  本帧涉及的 voxel 索引表（方便后续 recut/margi）
 * @param wdsize        滑窗大小
 * @param pwld          当前帧点云的世界坐标
 * @param sws           当前线程可用的 SlideWindow 池
 */
void cut_voxel(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map, int wdsize, PLV(3) &pwld, vector<SlideWindow*> &sws)
{
  int plsize = pvec->size();
  for(int i=0; i<plsize; i++)
  {
    pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    float loc[3];
    for(int j=0; j<3; j++)
    {
      loc[j] = pw[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      // voxel 已存在，直接把该点分配到对应 OctoTree/滑窗槽位
      iter->second->allocate(win_count, pv, pw, sws);
      iter->second->isexist = true;
      if(feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
    }
    else
    {
      // 新 voxel，创建 OctoTree 并设置中心/半径
      OctoTree* ot = new OctoTree(0, wdsize);
      ot->allocate(win_count, pv, pw, sws);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }
  }

}

// Cut the current scan into corresponding voxel in multi thread
/**
 * @brief 将当前帧点云按 voxel 划分并分配到 OctoTree，使用多线程加速写入滑窗
 * 
 * @param feat_map      全局 voxel 地图
 * @param pvec          当前帧点云（IMU系，带协方差）
 * @param win_count     当前帧在滑窗中的序号
 * @param feat_tem_map  本帧涉及的 voxel 集合（供后续 recut/margi）
 * @param wdsize        滑窗大小
 * @param pwld          当前帧点云的世界坐标
 * @param sws           多线程的 SlideWindow 池，按线程分片复用
 */
void cut_voxel_multi(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map, int wdsize, PLV(3) &pwld, vector<vector<SlideWindow*>> &sws)
{
  // 先分桶：按 voxel 归类所有点的索引，避免在临界区逐点分配
  unordered_map<OctoTree*, vector<int>> map_pvec;
  int plsize = pvec->size();
  for(int i=0; i<plsize; i++)
  {
    pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    float loc[3];
    for(int j=0; j<3; j++)
    {
      // loc[j] = pv.world[j] / voxel_size;
      loc[j] = pw[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    OctoTree* ot = nullptr;
    if(iter != feat_map.end())
    {
      // 已有 voxel，标记存在并加入本帧更新表
      iter->second->isexist = true;
      if(feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
      ot = iter->second;
    }
    else
    {
      // 新 voxel，创建 OctoTree 并写入地图
      ot = new OctoTree(0, wdsize);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }

    map_pvec[ot].push_back(i);
  }

  // for(auto iter=map_pvec.begin(); iter!=map_pvec.end(); iter++)
  // {
  //   for(int i: iter->second)
  //   {
  //     iter->first->allocate(win_count, (*pvec)[i], pwld[i], sws);
  //   }
  // }

  vector<pair<OctoTree *const, vector<int>>*> octs; octs.reserve(map_pvec.size());
  for(auto iter=map_pvec.begin(); iter!=map_pvec.end(); iter++)
    octs.push_back(&(*iter));

  int thd_num = sws.size();
  int g_size = octs.size();
  // voxel 数太少时不必多线程
  if(g_size < thd_num) return;
  vector<thread*> mthreads(thd_num);
  double part = 1.0 * g_size / thd_num;

  // 分块前，把滑窗池平均分发到各线程，降低锁竞争
  int swsize = sws[0].size() / thd_num;
  for(int i=1; i<thd_num; i++)
  {
    sws[i].insert(sws[i].end(), sws[0].end() - swsize, sws[0].end());
    sws[0].erase(sws[0].end() - swsize, sws[0].end());
  }

  for(int i=1; i<thd_num; i++)
  {
    mthreads[i] = new thread
    (
      [&](int head, int tail, vector<SlideWindow*> &sw)
      {
        for(int j=head; j<tail; j++)
        {
          for(int k: octs[j]->second)
            octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sw);
        }
      }, part*i, part*(i+1), ref(sws[i])
    );
  }

  for(int i=0; i<thd_num; i++)
  {
    if(i == 0)
    {
      for(int j=0; j<int(part); j++)
        for(int k: octs[j]->second)
          octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sws[0]);
    }
    else
    {
      mthreads[i]->join();
      delete mthreads[i];
    }
    
  }

}

/**
 * @brief 往地图中添加固定点云
 * 
 * @param feat_map 
 * @param pvec 
 * @param wdsize 
 * @param jour 
 */
void cut_voxel(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVec &pvec, int wdsize, double jour)
{
  for(pointVar &pv: pvec)
  {
    float loc[3];
    for(int j=0; j<3; j++)
    {
      loc[j] = pv.pnt[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->allocate_fix(pv);
    }
    else
    {
      OctoTree* ot = new OctoTree(0, wdsize);
      ot->push_fix_novar(pv);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->jour = jour;
      feat_map[position] = ot;
    }
  }
  
}

// Match the point with the plane in the voxel map
/**
 * @brief 计算点落在哪个voxel中，并匹配点与该voxel的平面
 * 
 * @param feat_map 
 * @param wld 
 * @param pla 
 * @param var_wld 
 * @param sigma_d 
 * @param oc 
 * @return int 
 */
int match(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, Eigen::Vector3d &wld, Plane* &pla, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc)
{
  int flag = 0;

  float loc[3];
  for(int j=0; j<3; j++)
  {
    loc[j] = wld[j] / voxel_size;
    if(loc[j] < 0) loc[j] -= 1;
  }
  VOXEL_LOC position(loc[0], loc[1], loc[2]);
  auto iter = feat_map.find(position);
  if(iter != feat_map.end())
  {
    double max_prob = 0;
    flag = iter->second->match(wld, pla, max_prob, var_wld, sigma_d, oc);
    // iter->second->match_end(pv, pla, max_prob);
    if(flag && pla==nullptr)
    {
      printf("pla null max_prob: %lf %ld %ld %ld\n", max_prob, iter->first.x, iter->first.y, iter->first.z);
    }
  }

  return flag;
}

#endif
