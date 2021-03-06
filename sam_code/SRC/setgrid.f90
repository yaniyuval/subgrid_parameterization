subroutine setgrid

! Initialize vertical grid

use grid	

implicit none
	
integer k, kmax

open(8,file='./'//trim(case)//'/grd',status='old',form='formatted') 

do k=1,nz     
  read(8,fmt=*,end=111) z(k)
  kmax=k
end do
goto 222
111 do k=kmax+1,nz
  z(k)=z(k-1)+(z(k-1)-z(k-2))
end do
222 continue
close (8)
 	
dz = 0.5*(z(1)+z(2))

do k=2,nzm
   adzw(k) = (z(k)-z(k-1))/dz
end do
adzw(1) = 1.
adzw(nz) = adzw(nzm)
adz(1) = 1.
do k=2,nzm-1
   adz(k) = 0.5*(z(k+1)-z(k-1))/dz
end do
adz(nzm) = adzw(nzm)
zi(1) = 0.
do k=2,nz
   zi(k) = zi(k-1) + adz(k-1)*dz
end do

if(LES) then
  do k=1,nzm
     grdf_x(k) = dx**2/(adz(k)*dz)**2
     grdf_y(k) = dy**2/(adz(k)*dz)**2
     grdf_z(k) = 1.
  end do
else
  do k=1,nzm
     grdf_x(k) = min(16.,(dx/ravefactor)**2/(adz(k)*dz)**2)
     grdf_y(k) = min(16.,(dy/ravefactor)**2/(adz(k)*dz)**2)
     grdf_x(k) = 1.
     grdf_y(k) = 1.
     grdf_z(k) = 1.
  end do
end if
end
