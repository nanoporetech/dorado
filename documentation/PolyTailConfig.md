# Custom Poly Tail Configuration

Dorado supports estimation of Poly(A/T) tails for DNA (PCS AND PCB) and RNA samples. The default settings are optimized for non-interrupted Poly(A/T) sequences that occur at read ends.

Dorado also supports additional features that can be customized through a configuration file (described below):
* Custom primer sequence for cDNA tail estimation
* Clustering of interrupted Poly(A/T) tails
* Estimation of Poly(A/T) length in plasmids

## Poly(A/T) Reference Diagram

```
cDNA

5' ---- ADAPTER ---- FRONT_PRIMER ---- cDNA ---- poly(A) ---- RC(PRIMER_REAR) ---- 3'

OR

5' ---- ADAPTER ---- PRIMER_REAR ---- poly(T) ---- RC(cDNA) ---- RC(FRONT_PRIMER) ---- 3'
```

```
dRNA

3' ---- ADAPTER ---- poly(A) ---- RNA ---- 5'
```

```
Plasmid

5' ---- ADAPTER ---- DNA ---- FRONT_FLANK ---- poly(A) ---- REAR_FLANK --- DNA ---- 3'

OR

5' ---- ADAPTER ---- RC(DNA) ---- RC(REAR_FLANK) ---- poly(T) ---- RC(FRONT_FLANK) ---- RC(DNA) ---- 3'
```

## Configuration Format

The configuration file needs to be in the `toml` format.

```
[anchors]
primer_front = "ATCG"
primer_rear = "CGTA"
plasmid_front_flank = "CGATCG"
plasmid_rear_flank = "TGACTGC"

[tail]
tail_interrupt_length = 10
```

### Configuration Options

| Option | Description |
| -- | -- |
| primer_front | Front primer sequence for cDNA |
| primer_rear | Rear primer sequence for cDNA |
| plasmid_front_flank | Front flanking sequence of poly(A) in plasmid |
| plasmid_rear_flank | Rear flanking sequence of poly(A) in plasmid |
| tail_interrupt_length | Combine tails that are within this distance of each other (default 0, don't combine any) |
