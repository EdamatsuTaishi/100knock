#2-11_tr
!tr '\t' ' ' < popular-names.txt > output.txt

#sed
!sed 's/\t/ /g' popular-names.txt > output1.txt

#expand
!expand -t 1 popular-names.txt > output2.txt