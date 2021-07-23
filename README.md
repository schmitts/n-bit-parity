# Brian2 simulation of a temporal n-bit parity task

The task is to find out if the sum of 1s is even or odd (the parity) in a
temporal stream of bits.

Bits are send in one after the other with a fixed input delay.
The neuron is stimulated if there is a 1 in the stream.

The neuron's membrane potential is read out after all bits are streamed in
after waiting for an additional readout delay.

Quantities are defined without units first, but units are attached
when entering Brian2. Think of times in ms.

Implementation partially based on: https://github.com/IGITUGraz/LSM

# Howto

```shell
./nbp.py
sim_time=10000
n_bit=10
input_delay_per_bit=0.1
membrane_time_constant=1
refractory_period=0.05
plot=False
stim_interval=20.0
readout_delay=1.1
mismatched training examples: 1/400 [0.2%]
1: 1 times wrongly classified
mismatched test examples: 0/100 [0.0%]
```
