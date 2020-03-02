subroutine compress3D (f,nx,ny,nz,name, long_name, units, &
       	               savebin, dompi, rank, nsubdomains)


! Compress3D: Compresses a given 3D array into the byte-array
! and writes the latter into a file.

	implicit none
! Input:

integer nx,ny,nz
real f(nx,ny,nz)
character*(*) name,long_name,units
integer rank,rrr,ttt,irank,nsubdomains
logical savebin, dompi

! Local:

integer(2), allocatable :: byte(:)
real(4), allocatable :: byte4(:)
integer size,count

character(10) value_min(nz), value_max(nz)
character(7) form
integer int_fac, integer_max, integer_min
parameter (int_fac=2,integer_min=-32000, integer_max=32000)
!	parameter (int_fac=1,integer_min=-127, integer_max=127)
real f_max,f_min, f_max1, f_min1, scale
integer i,j,k,req


! Allocate byte array:

size=nx*ny*nz
if(savebin) then
  allocate (byte4(size))
else
  allocate (byte(size))
end if
count = 0

if(savebin) then	

  do k=1,nz
   do j=1,ny
    do i=1,nx
     count = count+1
     byte4(count) = f(i,j,k)
    end do
   end do
  end do

  if(rank.eq.0) then
             write(46) name,' ',long_name,' ',units
             write(46) (byte4(k),k=1,count)
  end if

  do irank = 1, nsubdomains-1
    call task_barrier()
    if(irank.eq.rank) then
      call task_bsend_float(0,byte4,count,irank)
    end if
    if(rank.eq.0) then
      call task_receive_float(byte4,count,req)
      call task_wait(req,rrr,ttt)
      write(46) (byte4(k),k=1,count)
    end if
  end do

  deallocate(byte4)


else

  form='(f10.4)'

  do k=1,nz

   f_max=-1.e30
   f_min= 1.e30
   do j=1,ny
    do i=1,nx
     f_max = max(f_max,f(i,j,k)) 	    
     f_min = min(f_min,f(i,j,k)) 	    
    end do
   end do
   if(dompi) then
     f_max1=f_max
     f_min1=f_min
     call task_max_real(f_max1,f_max,1)
     call task_min_real(f_min1,f_min,1)
   endif

   write(value_max(k),form) f_max
   write(value_min(k),form) f_min

   scale = float(integer_max-integer_min)/(f_max-f_min+1.e-20)

   do j=1,ny
    do i=1,nx
      count=count+1
      byte(count)= integer_min+scale*(f(i,j,k)-f_min)   
    end do
   end do

  end do ! k
        
  if(rank.eq.0) then
     write(46) name,' ',long_name,' ',units,' ',value_max,value_min
     write(46) (byte(k),k=1,count)
  end if

  do irank = 1, nsubdomains-1
    call task_barrier()
    if(irank.eq.rank) then
      call task_send_character(0,byte,int_fac*count,irank,req)
      call task_wait(req,rrr,ttt)
    end if
    if(rank.eq.0) then
      call task_receive_character(byte,int_fac*count,req)
      call task_wait(req,rrr,ttt)
      write(46) (byte(k),k=1,count)
    end if
  end do

  deallocate(byte)


end if ! savebin


call task_barrier()

end subroutine compress3D
	
