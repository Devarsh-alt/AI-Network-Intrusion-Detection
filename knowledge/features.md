# Flow Features

## What a network flow is

The system does not look at individual packets. It classifies flows. A flow is all packets exchanged between two endpoints in one connection, summarised as statistics. "Forward" (Fwd) means packets from the host that started the connection, and "backward" (Bwd) means the replies. The CICIDS2017 features were extracted from raw packet captures with the CICFlowMeter tool. The model uses all 77 numeric features that CICFlowMeter produces; the sections below describe the main groups.

## Port and timing features

Destination Port is the port the flow was sent to, which indicates the service (80 web, 21 FTP, 22 SSH, 443 HTTPS). Flow Duration is the length of the flow in microseconds. Flow Bytes/s and Flow Packets/s are the byte rate and packet rate of the flow; very high rates suggest flooding, and very low rates over a long duration suggest a slow DoS attack.

## Packet count features

Total Fwd Packets and Total Backward Packets count the packets in each direction. A flow with one or two forward packets and no reply is typical of a port scan.

## Packet size features

Total Length of Fwd Packets and Total Length of Bwd Packets are the total payload bytes in each direction. Fwd Packet Length Max, Min and Mean and Bwd Packet Length Max, Min and Mean describe individual packet sizes per direction. Packet Length Mean, Packet Length Std and Packet Length Variance describe packet sizes over the whole flow, and Average Packet Size is the mean size of all packets. Attack tools tend to send packets of very uniform size, so low variance can be a warning sign.

## TCP flag features

SYN Flag Count counts packets with the SYN flag, which opens a TCP connection. ACK Flag Count counts packets with the ACK flag, which acknowledges received data. Many SYNs without matching ACKs point to scanning or a SYN flood.

## Inter-arrival time (IAT) features

Flow IAT, Fwd IAT and Bwd IAT (each with Mean, Std, Max, Min and Total) measure the time gaps between consecutive packets in the flow or in one direction. Automated tools send packets at very regular intervals, and slow attacks leave long gaps, so these timing features help separate scripted attacks from human-driven traffic.

## TCP window and header features

Init_Win_bytes_forward and Init_Win_bytes_backward are the initial TCP window sizes announced by each side. They depend on the operating system and the tool that opened the connection, so attack tools often have a recognisable value. Fwd Header Length and Bwd Header Length are the total bytes used by packet headers in each direction, and min_seg_size_forward is the smallest segment size seen in the forward direction.

## Subflow, bulk, active and idle features

Subflow features (Subflow Fwd Packets, Subflow Fwd Bytes and the backward equivalents) count packets and bytes in sub-periods of a flow. Active Mean, Std, Max and Min measure how long the flow was active before going idle, and Idle Mean, Std, Max and Min measure how long it stayed idle. Long idle periods are typical of slow DoS attacks and botnet beaconing.

## Why Destination Port is a caveat

In CICIDS2017 each attack was aimed at a fixed port (for example FTP brute force at port 21), so a model could learn "which port" instead of "what behaviour". The feature selection comparison checks this by training the hand-picked feature set with and without Destination Port. Removing it changes the scores only slightly, so the model does not depend on the port alone. Destination Port still often appears in the explanation of a single alert, because for one flow it can be the value that differs most from normal traffic.
