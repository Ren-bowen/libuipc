
template <typename T>
__host__ __device__ void E(T& R, const T& mu, const T& lambda, const Eigen::Vector<T,9>& VecF)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
LaTeX expression:
//tex:$$R = \frac{\lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)^{2}}{2} - \mu \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + \frac{\mu \left(F(0,0)^{2} + F(0,1)^{2} + F(0,2)^{2} + F(1,0)^{2} + F(1,1)^{2} + F(1,2)^{2} + F(2,0)^{2} + F(2,1)^{2} + F(2,2)^{2} - 3\right)}{2} + \frac{\mu^{2}}{\lambda^{2}}$$

Symbol Name Mapping:
mu:
    -> {}
    -> Matrix([[mu]])
lambda:
    -> {}
    -> Matrix([[lambda]])
VecF:
    -> {'VecF(0)': 'F(0,0)', 'VecF(1)': 'F(1,0)', 'VecF(2)': 'F(2,0)', 'VecF(3)': 'F(0,1)', 'VecF(4)': 'F(1,1)', 'VecF(5)': 'F(2,1)', 'VecF(6)': 'F(0,2)', 'VecF(7)': 'F(1,2)', 'VecF(8)': 'F(2,2)'}
    -> Matrix([[F(0,0)], [F(1,0)], [F(2,0)], [F(0,1)], [F(1,1)], [F(2,1)], [F(0,2)], [F(1,2)], [F(2,2)]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = VecF(0)*VecF(4)*VecF(8) - VecF(0)*VecF(7)*VecF(5) - VecF(3)*VecF(1)*VecF(8) + VecF(3)*VecF(7)*VecF(2) + VecF(6)*VecF(1)*VecF(5) - VecF(6)*VecF(4)*VecF(2) - 1;
/* Simplified Expr */
R = (1.0/2.0)*lambda*std::pow(x0, 2) - mu*x0 + (1.0/2.0)*mu*(std::pow(VecF(0), 2) + std::pow(VecF(3), 2) + std::pow(VecF(6), 2) + std::pow(VecF(1), 2) + std::pow(VecF(4), 2) + std::pow(VecF(7), 2) + std::pow(VecF(2), 2) + std::pow(VecF(5), 2) + std::pow(VecF(8), 2) - 3) + std::pow(mu, 2)/std::pow(lambda, 2);
}
template <typename T>
__host__ __device__ void dEdVecF(Eigen::Vector<T,9>& R, const T& mu, const T& lambda, const Eigen::Vector<T,9>& VecF)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
LaTeX expression:
//tex:$$R = \left[\begin{matrix}F(0,0) \mu + \frac{\lambda \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)\\F(1,0) \mu + \frac{\lambda \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right)\\F(2,0) \mu + \frac{\lambda \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right)\\F(0,1) \mu + \frac{\lambda \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)\\F(1,1) \mu + \frac{\lambda \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right)\\F(2,1) \mu + \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right)\\F(0,2) \mu + \frac{\lambda \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right)\\F(1,2) \mu + \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right)\\F(2,2) \mu + \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right)}{2} - \mu \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right)\end{matrix}\right]$$

Symbol Name Mapping:
mu:
    -> {}
    -> Matrix([[mu]])
lambda:
    -> {}
    -> Matrix([[lambda]])
VecF:
    -> {'VecF(0)': 'F(0,0)', 'VecF(1)': 'F(1,0)', 'VecF(2)': 'F(2,0)', 'VecF(3)': 'F(0,1)', 'VecF(4)': 'F(1,1)', 'VecF(5)': 'F(2,1)', 'VecF(6)': 'F(0,2)', 'VecF(7)': 'F(1,2)', 'VecF(8)': 'F(2,2)'}
    -> Matrix([[F(0,0)], [F(1,0)], [F(2,0)], [F(0,1)], [F(1,1)], [F(2,1)], [F(0,2)], [F(1,2)], [F(2,2)]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = VecF(4)*VecF(8);
auto x1 = VecF(7)*VecF(5);
auto x2 = VecF(3)*VecF(8);
auto x3 = VecF(6)*VecF(4);
auto x4 = (1.0/2.0)*lambda*(VecF(0)*VecF(4)*VecF(8) - VecF(0)*x1 + VecF(3)*VecF(7)*VecF(2) + VecF(6)*VecF(1)*VecF(5) - VecF(1)*x2 - VecF(2)*x3 - 1);
auto x5 = VecF(3)*VecF(7);
auto x6 = VecF(1)*VecF(8);
auto x7 = VecF(0)*VecF(8);
auto x8 = VecF(6)*VecF(2);
auto x9 = VecF(0)*VecF(7);
auto x10 = VecF(1)*VecF(5);
auto x11 = VecF(4)*VecF(2);
auto x12 = VecF(0)*VecF(5);
auto x13 = VecF(0)*VecF(4);
auto x14 = VecF(3)*VecF(1);
/* Simplified Expr */
R(0) = VecF(0)*mu - mu*(x0 - x1) + x4*(2*x0 - 2*x1);
R(1) = VecF(1)*mu - mu*(VecF(6)*VecF(5) - x2) + x4*(2*VecF(6)*VecF(5) - 2*x2);
R(2) = VecF(2)*mu - mu*(-x3 + x5) + x4*(-2*x3 + 2*x5);
R(3) = VecF(3)*mu - mu*(VecF(7)*VecF(2) - x6) + x4*(2*VecF(7)*VecF(2) - 2*x6);
R(4) = VecF(4)*mu - mu*(x7 - x8) + x4*(2*x7 - 2*x8);
R(5) = VecF(5)*mu - mu*(VecF(6)*VecF(1) - x9) + x4*(2*VecF(6)*VecF(1) - 2*x9);
R(6) = VecF(6)*mu - mu*(x10 - x11) + x4*(2*x10 - 2*x11);
R(7) = VecF(7)*mu - mu*(VecF(3)*VecF(2) - x12) + x4*(2*VecF(3)*VecF(2) - 2*x12);
R(8) = VecF(8)*mu - mu*(x13 - x14) + x4*(2*x13 - 2*x14);
}
template <typename T>
__host__ __device__ void ddEddVecF(Eigen::Matrix<T,9,9>& R, const T& mu, const T& lambda, const Eigen::Vector<T,9>& VecF)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
LaTeX expression:
//tex:$$R = \left[\begin{matrix}\frac{\lambda \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} + \mu & \frac{\lambda \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} & \frac{\lambda \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} & \frac{\lambda \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} & F(2,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(2,2) \mu + \frac{\lambda \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} & - F(1,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(1,2) \mu + \frac{\lambda \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} & \frac{\lambda \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} & - F(2,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(2,1) \mu + \frac{\lambda \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2} & F(1,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(1,1) \mu + \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(2 F(1,1) F(2,2) - 2 F(1,2) F(2,1)\right)}{2}\\\frac{\lambda \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & \frac{\lambda \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right) \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right)}{2} + \mu & \frac{\lambda \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right) \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right)}{2} & - F(2,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(2,2) \mu + \frac{\lambda \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} & \frac{\lambda \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right) \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right)}{2} & F(0,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(0,2) \mu + \frac{\lambda \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right) \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right)}{2} & F(2,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(2,1) \mu + \frac{\lambda \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right) \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right)}{2} & \frac{\lambda \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right) \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right)}{2} & - F(0,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(0,1) \mu + \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(- 2 F(0,1) F(2,2) + 2 F(0,2) F(2,1)\right)}{2}\\\frac{\lambda \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & \frac{\lambda \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right) \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right)}{2} & \frac{\lambda \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right) \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right)}{2} + \mu & F(1,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(1,2) \mu + \frac{\lambda \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} & - F(0,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(0,2) \mu + \frac{\lambda \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right) \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right)}{2} & \frac{\lambda \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right) \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right)}{2} & - F(1,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(1,1) \mu + \frac{\lambda \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right) \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right)}{2} & F(0,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(0,1) \mu + \frac{\lambda \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right) \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right)}{2} & \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(2 F(0,1) F(1,2) - 2 F(0,2) F(1,1)\right)}{2}\\\frac{\lambda \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & - F(2,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(2,2) \mu + \frac{\lambda \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right) \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right)}{2} & F(1,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(1,2) \mu + \frac{\lambda \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right) \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right)}{2} & \frac{\lambda \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} + \mu & \frac{\lambda \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right) \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right)}{2} & \frac{\lambda \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right) \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right)}{2} & \frac{\lambda \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right) \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right)}{2} & F(2,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(2,0) \mu + \frac{\lambda \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right) \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right)}{2} & - F(1,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(1,0) \mu + \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(- 2 F(1,0) F(2,2) + 2 F(1,2) F(2,0)\right)}{2}\\F(2,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(2,2) \mu + \frac{\lambda \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & \frac{\lambda \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right) \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right)}{2} & - F(0,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(0,2) \mu + \frac{\lambda \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right) \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right)}{2} & \frac{\lambda \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} & \frac{\lambda \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right) \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right)}{2} + \mu & \frac{\lambda \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right) \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right)}{2} & - F(2,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(2,0) \mu + \frac{\lambda \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right) \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right)}{2} & \frac{\lambda \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right) \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right)}{2} & F(0,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(0,0) \mu + \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(2 F(0,0) F(2,2) - 2 F(0,2) F(2,0)\right)}{2}\\- F(1,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(1,2) \mu + \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & F(0,2) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(0,2) \mu + \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right)}{2} + \mu & F(1,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(1,0) \mu + \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right)}{2} & - F(0,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(0,0) \mu + \frac{\lambda \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right) \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right)}{2} & \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(- 2 F(0,0) F(1,2) + 2 F(0,2) F(1,0)\right)}{2}\\\frac{\lambda \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & F(2,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(2,1) \mu + \frac{\lambda \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right) \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right)}{2} & - F(1,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(1,1) \mu + \frac{\lambda \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right) \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right)}{2} & \frac{\lambda \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} & - F(2,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(2,0) \mu + \frac{\lambda \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right) \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right)}{2} & F(1,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(1,0) \mu + \frac{\lambda \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right) \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right)}{2} & \frac{\lambda \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right) \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right)}{2} + \mu & \frac{\lambda \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right) \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right)}{2} & \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(2 F(1,0) F(2,1) - 2 F(1,1) F(2,0)\right)}{2}\\- F(2,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(2,1) \mu + \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right)}{2} & F(0,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(0,1) \mu + \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right)}{2} & F(2,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(2,0) \mu + \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right)}{2} & - F(0,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(0,0) \mu + \frac{\lambda \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right) \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right)}{2} & \frac{\lambda \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right) \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right)}{2} + \mu & \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(- 2 F(0,0) F(2,1) + 2 F(0,1) F(2,0)\right)}{2}\\F(1,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(1,1) \mu + \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(F(1,1) F(2,2) - F(1,2) F(2,1)\right)}{2} & - F(0,1) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(0,1) \mu + \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(- F(0,1) F(2,2) + F(0,2) F(2,1)\right)}{2} & \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(F(0,1) F(1,2) - F(0,2) F(1,1)\right)}{2} & - F(1,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) + F(1,0) \mu + \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(- F(1,0) F(2,2) + F(1,2) F(2,0)\right)}{2} & F(0,0) \lambda \left(F(0,0) F(1,1) F(2,2) - F(0,0) F(1,2) F(2,1) - F(0,1) F(1,0) F(2,2) + F(0,1) F(1,2) F(2,0) + F(0,2) F(1,0) F(2,1) - F(0,2) F(1,1) F(2,0) - 1\right) - F(0,0) \mu + \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(F(0,0) F(2,2) - F(0,2) F(2,0)\right)}{2} & \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(- F(0,0) F(1,2) + F(0,2) F(1,0)\right)}{2} & \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(F(1,0) F(2,1) - F(1,1) F(2,0)\right)}{2} & \frac{\lambda \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right) \left(- F(0,0) F(2,1) + F(0,1) F(2,0)\right)}{2} & \frac{\lambda \left(F(0,0) F(1,1) - F(0,1) F(1,0)\right) \left(2 F(0,0) F(1,1) - 2 F(0,1) F(1,0)\right)}{2} + \mu\end{matrix}\right]$$

Symbol Name Mapping:
mu:
    -> {}
    -> Matrix([[mu]])
lambda:
    -> {}
    -> Matrix([[lambda]])
VecF:
    -> {'VecF(0)': 'F(0,0)', 'VecF(1)': 'F(1,0)', 'VecF(2)': 'F(2,0)', 'VecF(3)': 'F(0,1)', 'VecF(4)': 'F(1,1)', 'VecF(5)': 'F(2,1)', 'VecF(6)': 'F(0,2)', 'VecF(7)': 'F(1,2)', 'VecF(8)': 'F(2,2)'}
    -> Matrix([[F(0,0)], [F(1,0)], [F(2,0)], [F(0,1)], [F(1,1)], [F(2,1)], [F(0,2)], [F(1,2)], [F(2,2)]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = VecF(4)*VecF(8);
auto x1 = VecF(7)*VecF(5);
auto x2 = x0 - x1;
auto x3 = (1.0/2.0)*lambda;
auto x4 = x3*(2*x0 - 2*x1);
auto x5 = VecF(3)*VecF(8);
auto x6 = VecF(6)*VecF(5) - x5;
auto x7 = VecF(3)*VecF(7);
auto x8 = VecF(6)*VecF(4);
auto x9 = x7 - x8;
auto x10 = VecF(1)*VecF(8);
auto x11 = VecF(7)*VecF(2) - x10;
auto x12 = VecF(0)*VecF(8);
auto x13 = VecF(6)*VecF(2);
auto x14 = x12 - x13;
auto x15 = VecF(8)*mu;
auto x16 = lambda*(VecF(0)*VecF(4)*VecF(8) - VecF(0)*x1 + VecF(3)*VecF(7)*VecF(2) + VecF(6)*VecF(1)*VecF(5) - VecF(1)*x5 - VecF(2)*x8 - 1);
auto x17 = VecF(8)*x16;
auto x18 = -x15 + x17;
auto x19 = VecF(0)*VecF(7);
auto x20 = VecF(6)*VecF(1) - x19;
auto x21 = VecF(7)*mu;
auto x22 = VecF(7)*x16;
auto x23 = x21 - x22;
auto x24 = VecF(1)*VecF(5);
auto x25 = VecF(4)*VecF(2);
auto x26 = x24 - x25;
auto x27 = VecF(0)*VecF(5);
auto x28 = VecF(3)*VecF(2) - x27;
auto x29 = VecF(5)*mu;
auto x30 = VecF(5)*x16;
auto x31 = x29 - x30;
auto x32 = VecF(0)*VecF(4);
auto x33 = VecF(3)*VecF(1);
auto x34 = x32 - x33;
auto x35 = VecF(4)*mu;
auto x36 = VecF(4)*x16;
auto x37 = -x35 + x36;
auto x38 = x3*(2*VecF(6)*VecF(5) - 2*x5);
auto x39 = x15 - x17;
auto x40 = VecF(6)*mu;
auto x41 = VecF(6)*x16;
auto x42 = -x40 + x41;
auto x43 = -x29 + x30;
auto x44 = VecF(3)*mu;
auto x45 = VecF(3)*x16;
auto x46 = x44 - x45;
auto x47 = x3*(2*x7 - 2*x8);
auto x48 = -x21 + x22;
auto x49 = x40 - x41;
auto x50 = x35 - x36;
auto x51 = -x44 + x45;
auto x52 = x3*(2*VecF(7)*VecF(2) - 2*x10);
auto x53 = VecF(2)*mu;
auto x54 = VecF(2)*x16;
auto x55 = -x53 + x54;
auto x56 = VecF(1)*mu;
auto x57 = VecF(1)*x16;
auto x58 = x56 - x57;
auto x59 = x3*(2*x12 - 2*x13);
auto x60 = x53 - x54;
auto x61 = VecF(0)*mu;
auto x62 = VecF(0)*x16;
auto x63 = -x61 + x62;
auto x64 = x3*(2*VecF(6)*VecF(1) - 2*x19);
auto x65 = -x56 + x57;
auto x66 = x61 - x62;
auto x67 = x3*(2*x24 - 2*x25);
auto x68 = x3*(2*VecF(3)*VecF(2) - 2*x27);
auto x69 = x3*(2*x32 - 2*x33);
/* Simplified Expr */
R(0,0) = mu + x2*x4;
R(0,1) = x4*x6;
R(0,2) = x4*x9;
R(0,3) = x11*x4;
R(0,4) = x14*x4 + x18;
R(0,5) = x20*x4 + x23;
R(0,6) = x26*x4;
R(0,7) = x28*x4 + x31;
R(0,8) = x34*x4 + x37;
R(1,0) = x2*x38;
R(1,1) = mu + x38*x6;
R(1,2) = x38*x9;
R(1,3) = x11*x38 + x39;
R(1,4) = x14*x38;
R(1,5) = x20*x38 + x42;
R(1,6) = x26*x38 + x43;
R(1,7) = x28*x38;
R(1,8) = x34*x38 + x46;
R(2,0) = x2*x47;
R(2,1) = x47*x6;
R(2,2) = mu + x47*x9;
R(2,3) = x11*x47 + x48;
R(2,4) = x14*x47 + x49;
R(2,5) = x20*x47;
R(2,6) = x26*x47 + x50;
R(2,7) = x28*x47 + x51;
R(2,8) = x34*x47;
R(3,0) = x2*x52;
R(3,1) = x39 + x52*x6;
R(3,2) = x48 + x52*x9;
R(3,3) = mu + x11*x52;
R(3,4) = x14*x52;
R(3,5) = x20*x52;
R(3,6) = x26*x52;
R(3,7) = x28*x52 + x55;
R(3,8) = x34*x52 + x58;
R(4,0) = x18 + x2*x59;
R(4,1) = x59*x6;
R(4,2) = x49 + x59*x9;
R(4,3) = x11*x59;
R(4,4) = mu + x14*x59;
R(4,5) = x20*x59;
R(4,6) = x26*x59 + x60;
R(4,7) = x28*x59;
R(4,8) = x34*x59 + x63;
R(5,0) = x2*x64 + x23;
R(5,1) = x42 + x6*x64;
R(5,2) = x64*x9;
R(5,3) = x11*x64;
R(5,4) = x14*x64;
R(5,5) = mu + x20*x64;
R(5,6) = x26*x64 + x65;
R(5,7) = x28*x64 + x66;
R(5,8) = x34*x64;
R(6,0) = x2*x67;
R(6,1) = x43 + x6*x67;
R(6,2) = x50 + x67*x9;
R(6,3) = x11*x67;
R(6,4) = x14*x67 + x60;
R(6,5) = x20*x67 + x65;
R(6,6) = mu + x26*x67;
R(6,7) = x28*x67;
R(6,8) = x34*x67;
R(7,0) = x2*x68 + x31;
R(7,1) = x6*x68;
R(7,2) = x51 + x68*x9;
R(7,3) = x11*x68 + x55;
R(7,4) = x14*x68;
R(7,5) = x20*x68 + x66;
R(7,6) = x26*x68;
R(7,7) = mu + x28*x68;
R(7,8) = x34*x68;
R(8,0) = x2*x69 + x37;
R(8,1) = x46 + x6*x69;
R(8,2) = x69*x9;
R(8,3) = x11*x69 + x58;
R(8,4) = x14*x69 + x63;
R(8,5) = x20*x69;
R(8,6) = x26*x69;
R(8,7) = x28*x69;
R(8,8) = mu + x34*x69;
}