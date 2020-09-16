scaProcessMSA -a Inputs/sh3_59.fasta -s source/data/2VKN -c A -p 0.3 0.2 0.2 0.8
scaCore -i output/sh3_59.db
scaSectorID -i output/sh3_59.db

mv output/sh3_59.db Inputs
rm -r output