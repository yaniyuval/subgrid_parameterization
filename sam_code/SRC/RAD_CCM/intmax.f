      integer function intmax(n,ix,inc)
c
c $Id: intmax.f,v 1.1.1.1 2007/01/11 22:58:18 cw Exp $
c $Author: cw $
c
      implicit none
      integer n,inc
      integer ix(*)
c
      integer i,mx
c
      mx = ix(1)
      intmax = 1
      do i=1+inc,inc*n,inc
         if (ix(i).gt.mx) then
            mx = ix(i)
            intmax = i
         end if
      end do
      return
      end
 
