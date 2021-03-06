function de = dtesatw(t)
% t is temperature (K)
% from sat.f90

a0 = 0.443956472;
a1 = 0.285976452e-1;
a2 = 0.794747212e-3;
a3 = 0.121167162e-4;
a4 = 0.103167413e-6;
a5 = 0.385208005e-9;
a6 = -0.604119582e-12;
a7 = -0.792933209e-14;
a8 = -0.599634321e-17;

dt = max(-80.,t-273.16);

de = a0 + dt.*(a1+dt.*(a2+dt.*(a3+dt.*(a4+dt.*(a5+dt.*(a6+dt.*(a7+a8.*dt)))))));

