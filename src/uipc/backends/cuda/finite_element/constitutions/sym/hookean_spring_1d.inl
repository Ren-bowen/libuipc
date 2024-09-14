
template <typename T>
__host__ __device__ void E(T& R, const T& k, const Eigen::Vector<T,6>& X, const T& L0)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
Symbol Name Mapping:
k:
    -> {}
    -> Matrix([[k]])
X:
    -> {}
    -> Matrix([[X(0)], [X(1)], [X(2)], [X(3)], [X(4)], [X(5)]])
L0:
    -> {}
    -> Matrix([[L0]])
*****************************************************************************************************************************/
/* Sub Exprs */
/* Simplified Expr */
R = (1.0/2.0)*k*std::pow((-L0 + std::pow((std::pow((-X(0) + X(3)), (2)) + std::pow((-X(1) + X(4)), (2)) + std::pow((-X(2) + X(5)), (2))), (1.0/2.0))), (2))/std::pow((L0), (2));
}
template <typename T>
__host__ __device__ void dEdX(Eigen::Vector<T,6>& R, const T& k, const Eigen::Vector<T,6>& X, const T& L0)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
Symbol Name Mapping:
k:
    -> {}
    -> Matrix([[k]])
X:
    -> {}
    -> Matrix([[X(0)], [X(1)], [X(2)], [X(3)], [X(4)], [X(5)]])
L0:
    -> {}
    -> Matrix([[L0]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = X(0) - X(3);
auto x1 = -x0;
auto x2 = X(1) - X(4);
auto x3 = -x2;
auto x4 = X(2) - X(5);
auto x5 = -x4;
auto x6 = std::pow((std::pow((x1), (2)) + std::pow((x3), (2)) + std::pow((x5), (2))), (1.0/2.0));
auto x7 = k*(-L0 + x6)/(std::pow((L0), (2))*x6);
/* Simplified Expr */
R(0) = x0*x7;
R(1) = x2*x7;
R(2) = x4*x7;
R(3) = x1*x7;
R(4) = x3*x7;
R(5) = x5*x7;
}
template <typename T>
__host__ __device__ void ddEddX(Eigen::Matrix<T,6,6>& R, const T& k, const Eigen::Vector<T,6>& X, const T& L0)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
Symbol Name Mapping:
k:
    -> {}
    -> Matrix([[k]])
X:
    -> {}
    -> Matrix([[X(0)], [X(1)], [X(2)], [X(3)], [X(4)], [X(5)]])
L0:
    -> {}
    -> Matrix([[L0]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = X(0) - X(3);
auto x1 = std::pow((x0), (2));
auto x2 = -x0;
auto x3 = std::pow((x2), (2));
auto x4 = X(1) - X(4);
auto x5 = -x4;
auto x6 = std::pow((x5), (2));
auto x7 = X(2) - X(5);
auto x8 = -x7;
auto x9 = std::pow((x8), (2));
auto x10 = x3 + x6 + x9;
auto x11 = k/std::pow((L0), (2));
auto x12 = x11/x10;
auto x13 = std::pow((x10), (1.0/2.0));
auto x14 = x11*(-L0 + x13);
auto x15 = x14/x13;
auto x16 = x14/std::pow((x10), (3.0/2.0));
auto x17 = x0*x16;
auto x18 = x15 + x17*x2;
auto x19 = x0*x12;
auto x20 = x19*x4;
auto x21 = x17*x5;
auto x22 = x19*x7;
auto x23 = x17*x8;
auto x24 = -x15;
auto x25 = x19*x2 + x24;
auto x26 = x19*x5;
auto x27 = x17*x4;
auto x28 = x19*x8;
auto x29 = x17*x7;
auto x30 = x16*x4;
auto x31 = x2*x30;
auto x32 = std::pow((x4), (2));
auto x33 = x15 + x30*x5;
auto x34 = x12*x4;
auto x35 = x34*x7;
auto x36 = x30*x8;
auto x37 = x2*x34;
auto x38 = x24 + x34*x5;
auto x39 = x34*x8;
auto x40 = x30*x7;
auto x41 = x16*x7;
auto x42 = x2*x41;
auto x43 = x41*x5;
auto x44 = std::pow((x7), (2));
auto x45 = x15 + x41*x8;
auto x46 = x12*x7;
auto x47 = x2*x46;
auto x48 = x46*x5;
auto x49 = x24 + x46*x8;
auto x50 = x16*x2;
auto x51 = x5*x50;
auto x52 = x50*x8;
auto x53 = x12*x2;
auto x54 = x5*x53;
auto x55 = x53*x8;
auto x56 = x5*x8;
auto x57 = x16*x56;
auto x58 = x12*x56;
/* Simplified Expr */
R(0,0) = x1*x12 + x18;
R(0,1) = x20 + x21;
R(0,2) = x22 + x23;
R(0,3) = x1*x16 + x25;
R(0,4) = x26 + x27;
R(0,5) = x28 + x29;
R(1,0) = x20 + x31;
R(1,1) = x12*x32 + x33;
R(1,2) = x35 + x36;
R(1,3) = x27 + x37;
R(1,4) = x16*x32 + x38;
R(1,5) = x39 + x40;
R(2,0) = x22 + x42;
R(2,1) = x35 + x43;
R(2,2) = x12*x44 + x45;
R(2,3) = x29 + x47;
R(2,4) = x40 + x48;
R(2,5) = x16*x44 + x49;
R(3,0) = x16*x3 + x25;
R(3,1) = x37 + x51;
R(3,2) = x47 + x52;
R(3,3) = x12*x3 + x18;
R(3,4) = x31 + x54;
R(3,5) = x42 + x55;
R(4,0) = x26 + x51;
R(4,1) = x16*x6 + x38;
R(4,2) = x48 + x57;
R(4,3) = x21 + x54;
R(4,4) = x12*x6 + x33;
R(4,5) = x43 + x58;
R(5,0) = x28 + x52;
R(5,1) = x39 + x57;
R(5,2) = x16*x9 + x49;
R(5,3) = x23 + x55;
R(5,4) = x36 + x58;
R(5,5) = x12*x9 + x45;
}