rm -rf log
make -j4 -C ..



mkdir -p model
./train pos_list.txt neg_list.txt model.dat
#gdb --args ./train pos_list.txt neg_list.txt model.dat


