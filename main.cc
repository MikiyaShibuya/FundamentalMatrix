#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <set>

constexpr int seed = 1;

using Mat33_t = Eigen::Matrix3d;
using Vec3_t = Eigen::Vector3d;
using Vec2_t = Eigen::Vector2d;

template<typename T>
using eigen_vector = std::vector<T, Eigen::aligned_allocator<T>>;

eigen_vector<Vec3_t> gen_3Dpoints(const int n) {

    std::mt19937 mt(seed);
    const double z_dist = 3;
    const double r = 1;
    std::uniform_real_distribution<> rand(-r, r);

    eigen_vector<Vec3_t> pts;
    pts.reserve(n);
    for (unsigned int i = 0; i < n; i++) {
        const double x = rand(mt);
        const double y = rand(mt);
        const double z = rand(mt) + z_dist;
        pts.emplace_back(x, y, z);
    }
    return pts;
}

template<typename T>
void generate_keypts(const int num_pts, const double outlier_ratio,
                     T &kps_1, T &kps_2, std::vector<bool> &is_inliers) {

    double data[9];
    data[0] = 0.99875035875531126;
    data[1] = 0.043232155011259037;
    data[2] = 0.025073923889563768;
    data[3] = -0.043243275321131029;
    data[4] = 0.9990645671709939;
    data[5] = -9.8807568748168046e-05;
    data[6] = -0.025054740582133875;
    data[7] = -0.00098559449940248617;
    data[8] = 0.99968559486362751;

    Mat33_t R(data);
    Vec3_t t{-0.49, 0.12, -0.05};

    eigen_vector<Vec3_t> pts = gen_3Dpoints(num_pts);

    Mat33_t K;
    K << 500, 0, 500,
            0, 500, 250,
            0, 0, 1;

    kps_1.clear();
    kps_1.reserve(num_pts);
    kps_2.clear();
    kps_2.reserve(num_pts);

    std::mt19937 mt(seed);
    const double r = 1;
    std::uniform_real_distribution<> rand(-r, r);
    const auto try_bernoulli = [&mt](const double p) {
        std::uniform_real_distribution<> rand(0, 1);
        return rand(mt) < p;
    };

    for (const auto &pos_w : pts) {

        const double x = rand(mt);
        const double y = rand(mt);
        const double z = rand(mt);
        const Vec3_t noise{x, y, z};

        const bool is_outlier = try_bernoulli(outlier_ratio);


        const Vec3_t &pos_1 = pos_w;
        const Vec3_t pos_2 = R * pos_w + t
                             + (is_outlier ? noise : Vec3_t::Zero());

        kps_1.emplace_back((K * pos_1).hnormalized());
        kps_2.emplace_back((K * pos_2).hnormalized());
        is_inliers.push_back(!is_outlier);
    }
}

// Fundamental solver; Modified version of eightpt solver from opengv
// Reference:
// https://github.com/laurentkneip/opengv/blob/91f4b19c73450833a40e463ad3648aae80b3a7f3/src/relative_pose/methods.cpp#L424
template<typename T>
Mat33_t solve_F(const T &kps_1, const T &kps_2) {
    const int num_pts = kps_1.size();
    Eigen::MatrixXd A(num_pts, 9);

    for (unsigned int i = 0; i < num_pts; i++) {
        A.block<1, 3>(i, 0) = kps_2.at(i)(0) * kps_1.at(i).homogeneous();
        A.block<1, 3>(i, 3) = kps_2.at(i)(1) * kps_1.at(i).homogeneous();
        A.block<1, 3>(i, 6) = kps_1.at(i).homogeneous();
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> SVD(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix<Mat33_t::Scalar, 9, 1> f = SVD.matrixV().col(8);
    Mat33_t F_temp(3, 3);
    F_temp.row(0) = f.block<3, 1>(0, 0).transpose();
    F_temp.row(1) = f.block<3, 1>(3, 0).transpose();
    F_temp.row(2) = f.block<3, 1>(6, 0).transpose();

    const Eigen::JacobiSVD<Mat33_t> SVD2(F_temp, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Vec3_t s = SVD2.singularValues();
    s(2) = 0.0;

    const Mat33_t &U = SVD2.matrixU();
    const Mat33_t S = s.asDiagonal();
    const Mat33_t &V_T = SVD2.matrixV().transpose();

    return U * S * V_T;
}

template<typename T>
std::vector<bool> check_inlier(const Mat33_t &F, const T &kps_1, const T &kps_2) {
    std::vector<bool> inl_flag(kps_1.size(), false);
    for (int i = 0; i < kps_1.size(); i++) {
        const double err = kps_2.at(i).homogeneous().dot(F * kps_1.at(i).homogeneous());
        inl_flag.at(i) = std::abs(err) < 1e-8;
    }
    return inl_flag;
}


// Find F matrix by ransac
template<typename T>
Mat33_t estimate_F(const T &kps_1, const T &kps_2, const int num_trial) {
    const auto num_pts = kps_1.size();
    const auto num_sample = std::min<int>(8, num_pts);

    std::uniform_int_distribution<> dist(0, num_pts - 1);
    std::mt19937 mt(seed);

    Mat33_t F;
    int max_inl = 0;

    for (unsigned int i = 0; i < num_trial; i++) {
        std::set<int> sample_id;
        while (sample_id.size() < num_sample) {
            sample_id.insert(dist(mt));
        }

        T smpl1, smpl2;
        for (const auto id: sample_id) {
            smpl1.push_back(kps_1.at(id));
            smpl2.push_back(kps_2.at(id));
        }
        const Mat33_t F_cand = solve_F(smpl1, smpl2);
        const auto inl_flag = check_inlier(F_cand, kps_1, kps_2);
        const int num_inl = std::count(inl_flag.begin(), inl_flag.end(), true);
        if (max_inl < num_inl) {
            max_inl = num_inl;
            F = F_cand;
        }
    }

    return F;
}

template<typename T>
void print_test_result(const T &is_inliers, const T &inl_flag) {
    int true_pos = 0;
    int true_neg = 0;
    int false_pos = 0;
    int false_neg = 0;

    const auto num_pts = is_inliers.size();
    for (int i = 0; i < num_pts; i++) {
        const bool gt = is_inliers.at(i);
        const bool est = inl_flag.at(i);
        if (gt == est) {
            if (gt) true_pos++;
            else true_neg++;
        } else {
            if (est) false_pos++;
            else false_neg++;
        }
    }

    std::cout << std::endl;
    std::cout << "<< Confusion Matrix >>" << std::endl;
    std::cout << " GT \\ est | inlier | outlier |" << std::endl;

    std::cout << "  inlier  |  ";
    std::cout << std::setw(4) << true_pos << "  |  ";
    std::cout << std::setw(4) << false_neg << "   |" << std::endl;

    std::cout << " outlier  |  ";
    std::cout << std::setw(4) << false_pos << "  |  ";
    std::cout << std::setw(4) << true_neg << "   |" << std::endl;
}

int main() {
    const int num_pts = 1000;
    const double outlier_ratio = 0.3;

    // テスト用の特徴点と外れ値かどうかの正解
    eigen_vector<Vec2_t> kps_1, kps_2;
    std::vector<bool> is_inliers;
    generate_keypts(num_pts, outlier_ratio, kps_1, kps_2, is_inliers);

    // F行列を推定
    // num_trial は (1/(1-outlier_ratio)) ^ 8 より大きく
    const Mat33_t F = estimate_F(kps_1, kps_2, 100);

    // 推定したF行列を用いて特徴点マッチのインライア判定（インデックスでマッチ）
    const auto inl_flag = check_inlier(F, kps_1, kps_2);

    // F行列を用いたインライア判定の結果の表示
    const auto num_inls = std::count(inl_flag.begin(), inl_flag.end(), true);
    std::cout << "#inlier: " << num_inls << std::endl;

    // F行列推定による外れ値検出の正当性の表示
    print_test_result(is_inliers, inl_flag);
}
