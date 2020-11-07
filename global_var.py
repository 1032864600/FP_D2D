global Ksize
# K: users; L: IRSs; M1: elements of IRS; M: receiver antennas; N: transmitter antennas
# Ksize = [K, L, Ml, M, N]
Ksize = [4, 1, 100, 1, 1]

global train_batch
train_batch = int(5e2)

global val_batch
val_batch = int(5e2)

global train_sample_num
train_sample_num = train_batch*400

global val_sample_num
val_sample_num = val_batch*10