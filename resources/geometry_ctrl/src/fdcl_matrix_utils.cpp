#include "fdcl_matrix_utils.hpp"

Matrix3 hat(const Vector3 v){
  Matrix3 V;
  V.setZero();

  V(2,1) = v(0);
  V(1,2) = -V(2, 1);
  V(0,2) = v(1);
  V(2,0) = -V(0, 2);
  V(1,0) = v(2);
  V(0,1) = -V(1, 0);

  return V;
}

Vector3 vee(const Matrix3 V){
  Vector3 v;
  Matrix3 E;

  v.setZero();
  E = V + V.transpose();
    
  if(E.norm() > 1.e-6){std::cout << "VEE: E.norm() = " << E.norm() << std::endl;}

  v(0) = V(2, 1);
  v(1) = V(0, 2);
  v(2) = V(1, 0);

  return  v;
}

void saturate(Vector3 &x, const double x_min, const double x_max){
  for (int i = 0; i < 3; i++){
    if (x(i) > x_max){x(i) = x_max;}
    else if (x(i) < x_min){x(i) = x_min;}
  }
}

void deriv_unit_vector(const Vector3& A, const Vector3& A_dot, const Vector3& A_ddot, Vector3& q, Vector3& q_dot, Vector3& q_ddot) {
  constexpr double eps2 = 0.05;

  const double nA2 = A.squaredNorm();
  if (nA2 < eps2) {
    std::cout << "DERIV_UNIT_VECTOR: A.norm() = " << nA2 << std::endl;
    return;
  }

  const double nA = std::sqrt(nA2);
  const double inv_nA = 1.0 / nA;
  const double inv_nA3 = inv_nA / nA2;
  const double inv_nA5 = inv_nA3 / nA2;

  const double Adot_A = A.dot(A_dot);
  const double Adot_Adot = A_dot.dot(A_dot);
  const double A_Addot = A.dot(A_ddot);

  q = A * inv_nA;

  q_dot = A_dot * inv_nA
        - A * (Adot_A * inv_nA3);

  q_ddot = A_ddot * inv_nA
         - A_dot * (2.0 * Adot_A * inv_nA3)
         - A * ((Adot_Adot + A_Addot) * inv_nA3)
         + A * (3.0 * Adot_A * Adot_A * inv_nA5);
}