# Custom Poly Tail Configuration

Dorado supports estimation of Poly(A/T) tails for DNA (PCS AND PCB) and RNA samples. The default settings are optimized for non-interrupted Poly(A/T) sequences that occur at read ends.

Dorado also supports additional features that can be customized through a configuration file (described below):
* Custom primer sequence for cDNA tail estimation
* Clustering of interrupted Poly(A/T) tails

## Poly(A/T) Reference Diagram

```
cDNA

5' ---- ADAPTER ---- FRONT_PRIMER ---- cDNA ---- poly(A) ---- RC(REAR_PRIMER) ---- 3'

OR

5' ---- ADAPTER ---- REAR_PRIMER ---- poly(T) ---- RC(cDNA) ---- RC(FRONT_PRIMER) ---- 3'
```

```
dRNA

3' ---- ADAPTER ---- poly(A) ---- RNA ---- 5'
```

## Configuration Format

The configuration file needs to be in the `toml` format.

```
[anchors]
front_primer = "ATCG"
rear_primer = "CGTA"

[threshold]
flank_threshold = 10

[tail]
tail_interrupt_length = 10
```

### Configuration Options

| Option | Description |
| -- | -- |
| front_primer | Front primer sequence for cDNA |
| rear_primer | Rear primer sequence for cDNA |
| flank_threshold  | The edit distance threshold to use for detection of the flank/primer sequences |
| tail_interrupt_length | Combine tails that are within this distance of each other (default is 0, i.e. don't combine any) |
