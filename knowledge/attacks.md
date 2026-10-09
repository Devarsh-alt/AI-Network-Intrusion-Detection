# Attack Types

## Denial of Service (DoS and DDoS)

A Denial of Service attack floods a server with traffic or holds its connections open so that legitimate users cannot be served. A Distributed DoS (DDoS) does the same from many machines at once. In CICIDS2017 this category groups the labels DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest and DDoS (generated with the LOIC tool).

How it looks in flow features: volumetric attacks such as Hulk and DDoS produce very many flows to one web port (usually 80) with high Flow Packets/s and large backward packet lengths. Slow attacks such as slowloris and Slowhttptest do the opposite: very long Flow Duration with few, small packets, because the attacker keeps connections open as long as possible.

Response: rate-limit or block the offending sources, enable SYN cookies and connection limits on the server, and put a reverse proxy, CDN or DDoS scrubbing service in front of public services.

## Port Scan

A port scan probes many ports on a host to find which services are open. It is reconnaissance: it usually comes before an actual attack. CICIDS2017 port scans were generated with Nmap.

How it looks in flow features: a huge number of very short flows to many different Destination Port values, each with one or two packets, almost no payload (tiny packet lengths) and little or no response from the target.

Response: it is rated Medium severity because no damage is done yet. Block the scanning source at the firewall, close or filter ports that do not need to be exposed, and watch the same source for follow-up attacks.

## Brute Force

A brute-force attack tries many username and password combinations against a login service. In CICIDS2017 this covers FTP-Patator (port 21) and SSH-Patator (port 22), both generated with the Patator tool.

How it looks in flow features: many repeated flows to port 21 or 22 with similar small packet sizes and similar durations, since each flow is one failed login attempt.

Response: lock accounts or add delays after repeated failures (for example fail2ban), require key-based SSH authentication or multi-factor authentication, and restrict management ports to trusted networks.

## Botnet

A botnet is a group of infected machines controlled remotely by an attacker through a command-and-control (C2) server. The CICIDS2017 Bot traffic was generated with the Ares botnet.

How it looks in flow features: infected hosts send periodic, low-volume "beacon" flows to the C2 server. These look a lot like normal traffic, which makes Botnet one of the hardest categories for the model: a noticeable share of botnet flows is classified as normal.

Response: it is rated Critical because it means a machine inside the network is already compromised. Isolate the host, block the C2 destination, and reimage or clean the machine.

## Web Attack

Web attacks target a web application rather than the network. CICIDS2017 includes web login brute force, Cross-Site Scripting (XSS) and SQL Injection, all against port 80.

How it looks in flow features: the malicious part is in the HTTP payload, which flow statistics do not see. The model can only pick up side effects such as request sizes, timing between packets and TCP window sizes. With the 20 hand-picked features of the first version the model missed most web attacks; with all flow features it detects them well (see the feature selection comparison).

Response: validate and escape user input, use parameterised SQL queries, and deploy a Web Application Firewall (WAF).

## Other (Infiltration and Heartbleed)

This category holds the two rarest CICIDS2017 labels. Infiltration is an attack from inside the network after a victim opens a malicious file. Heartbleed exploits a bug in old OpenSSL versions (CVE-2014-0160) to read server memory through port 444 in this dataset.

There are only a few dozen flows of these in the whole dataset, so the model's scores for this category are based on very few test samples and should be treated with caution. Alerts in this category are rated Critical.

## NSL-KDD attack categories

NSL-KDD groups its attacks into four categories. DoS (for example neptune, smurf, back) makes a service unavailable. Probe (for example satan, ipsweep, nmap, portsweep) is scanning and reconnaissance. R2L, Remote to Local (for example guess_passwd, warezmaster), is an outside attacker gaining local access to a machine. U2R, User to Root (for example buffer_overflow, rootkit), is a local user escalating to administrator privileges. R2L and U2R are rare and look like normal sessions, so they are the hardest to detect.
