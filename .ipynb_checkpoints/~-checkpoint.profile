df -H ~ | tail -1 | awk '{print " Disk Space: " $3 " of " $2 " Used - PV Disk " $5 " Full\r\n"}'
