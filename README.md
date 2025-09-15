# multi_hop_industrial_simulator
This is a discrete-time network simulator, written in Python, that models a THz wireless network for an Industrial Internet of Things (IIoT) scenario in the context of the TIMES project. 
It is designed to evaluate the performance of multi-hop communication in a factory-like scenario, with a focus on the success probability, the average latency, the network throughput and the jain index. 
To enable multi-hop efficiently, three different routing algorithms have been implemented: 

1. Table-Based (TB) - it is the proposed solution, where route discovery and maintenance are performed through the user-plane data exhchanges only. TB is also extended with a Multi-Agent Deep Reinforcement Learning (MADRL) algorithm to autonomously adapt several Medium Access Control (MAC) parameters);
2. Table-Less (TL) - it is a broadcast-based protocol that forwards all received data without maintaining routing tables;
3. Ad hoc On-Demand Distance Vector (AODV) - it relies on control messages for neighbor and route discovery.

All routing algorithms operate on top of an unslotted Aloha MAC protocol, while for the wireless propagation, the simulator can use either an experimentally derived channel model, or  the 3GPP indoor factory channel model from TR 38.901.  

The simulation environment models an industrial scenario with the following characteristics:
- Base station (BS): A BS is deployed at the center of the area, serving as the anchor point for communication.
- User Equipment (UE) deployment:
    - UEs are placed over a regular grid layout to ensure that all of them can reach the BS through one or more hops. Note that, if the input parameters (e.g., UE transmit power, UE antenna gain etc.) are changed, it is necessary to modify their coordinates within the environment to ensure that they can reach the BS;
    - UEs can also be randomly and uniformly distributed in the environment, with the constraint of having at least a neighbor to reach the BS if they cannot do it directly.
- Obstacles: Fixed-position obstacles are included, which block some links and create Line-of-Sight (LOS) or Non-Line-of-Sight (NLOS) conditions.
- Full-buffer traffic model:
    - Each UE always has data to send.
    - As soon as a DATA packet is successfully delivered (or discarded after exceeding the maximum number of retransmission attempts), a new packet is immediately generated.  
These assumptions ensure that the network operates under saturated traffic conditions, highlighting the impact of routing strategies and channel impairments on performance.

Repository Structure

- env - Deployment of the reference scenario:
  - Creates the industrial layout (parallelepiped).
  - Positions UEs, BS, and machines (obstacles) at predefined locations and according to the selected distribution type.
- Network - Defines UEs and BS:
  - Initialization of entities.
  - Methods for interaction and simulation execution.
- Traffic_models
- Channel_models:
  - Path loss computation.
  - Received power calculation.
  - Absorption effects evaluation.
- Utils - Utility functions:
  - Interference and collision check at the receiver
  - LOS/NLOS condition evaluation.
  - Next-action logic for the RL-based algorithm.
  - Instantiation of UEs and BS.
- Test: main files for routing strategies:
  - TB,
  - TL,
  - AODV,
  - TB with MADRL.
- Results: Folder for storing simulation outputs.



The main reference for the simulator is the paper: "MAC and Routing Protocols Design in Multi-Hop Terahertz Networks", S. Cavallero, A. Pumilia, G. Cuozzo, C. Buratti....

