	subroutine write_rad()
	
	use rad
        use radae, only: abstot_3d, absnxt_3d, emstot_3d
	implicit none
	character *4 rankchar
	integer irank
        integer lenstr
        external lenstr


        write(rankchar,'(i4)') nsubdomains

        do irank=0,nsubdomains-1

           call task_barrier()

           if(irank.eq.rank) then

             if(rank.eq.0) then


               open(56,file='./RESTART/'//case(1:len_trim(case))//'_'// &
                    caseid(1:len_trim(caseid))//'_'// &
                    rankchar(5-lenstr(rankchar):4)//'_restart_rad.bin', &
                    status='unknown',form='unformatted')
                write(56) nsubdomains

             else

               open(56,file='./RESTART/'//case(1:len_trim(case))//'_'// & 
                    caseid(1:len_trim(caseid))//'_'// &
                    rankchar(5-lenstr(rankchar):4)//'_restart_rad.bin', &
                    status='unknown',form='unformatted', position='append')

             end if

	     write(56) initrad,nradsteps,tabs_rad,qc_rad,qi_rad,qv_rad, &
		 qrad,absnxt_3d,abstot_3d,emstot_3d 
             close(56)
	     if(rank.eq.0) then
               print *,'Saved radiation restart file. nstep=',nstep
	     endif
           end if
        end do

        call task_barrier()

        return
        end
 
 
 
 
     
	subroutine read_rad()
	
	use rad
        use radae, only: abstot_3d, absnxt_3d, emstot_3d
	implicit none
	character *4 rankchar
	integer irank,ii
        integer lenstr
        external lenstr
	
        write(rankchar,'(i4)') nsubdomains

        open(56,file='./RESTART/'//case(1:len_trim(case))//'_'// &
          caseid(1:len_trim(caseid))//'_'// &
          rankchar(5-lenstr(rankchar):4)//'_restart_rad.bin', &
          status='unknown',form='unformatted')

        do irank=0,nsubdomains-1

           call task_barrier()

           if(irank.eq.rank) then

             read (56)

             do ii=0,irank-1 ! skip records
               read(56)
             end do

	     read(56) initrad,nradsteps,tabs_rad,qc_rad,qi_rad,qv_rad, &
		 qrad,absnxt_3d,abstot_3d,emstot_3d 
             close(56)
             if(rank.eq.nsubdomains-1) then
                 print *,'Case:',caseid
                 print *,'Restart radiation at step:',nstep
                 print *,'Time:',nstep*dt
             endif

           end if

        end do

        call task_barrier()


        return
        end
