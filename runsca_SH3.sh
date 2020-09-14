if [ -e pySCA/data/2VKN.pdb ]; then
	echo ''
else
	cp Inputs/2VKN.pdb pySCA/data
fi

cd pySCA

if [ -e output ]; then
	echo ''
else
	mkdir output
fi

./pysca/scaProcessMSA.py ../Inputs/sh3_59.fasta -s 2VKN -c A -o 321
./pysca/scaCore.py output/sh3_59.db
./pysca/scaSectorID.py output/sh3_59.db

cp output/sh3_59.db ../Inputs
