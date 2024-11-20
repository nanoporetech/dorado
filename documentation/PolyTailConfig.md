# Custom Poly Tail Configuration

Dorado supports estimation of Poly(A/T) tails for DNA (PCS and PCB) and RNA samples. The default settings are optimized for non-interrupted Poly(A/T) sequences that occur at read ends.

Dorado also supports additional features that can be customized through a configuration file (described below):
* Custom primer sequence for cDNA tail estimation
* Clustering of interrupted Poly(A/T) tails
* Estimation of Poly(A/T) length in plasmids

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
front_primer = "ATCG"
rear_primer = "CGTA"
plasmid_front_flank = "CGATCG"
plasmid_rear_flank = "TGACTGC"

[threshold]
flank_threshold = 0.6

[tail]
tail_interrupt_length = 10
```

### Overrides
Configuration options can be overridden for individual barcodes. We generate a default configuration as normal, and then
add overrides of specific values for each barcode by adding an `[[overrides]]` section labelled by the barcode name.
```
[anchors]
front_primer = "ATCG"
rear_primer = "CGTA"
[tail]
tail_interrupt_length = 5

[[overrides]]
barcode_id = "Custom-Kit_barcode01"
[overrides.threshold]
flank_threshold = 0.5

[[overrides]]
barcode_id = "Custom-Kit_barcode02"
[overrides.anchors]
front_primer = "AACC"
rear_primer = "GGTT"
[overrides.tail]
tail_interrupt_length = 10
```

This creates three configurations:
* a default configuration with custom front and rear primers and an interrupt length of 5
* a configuration to use for `barcode01` from kit `Custom-Kit`  identical to the main custom settings (i.e. with the custom front and rear primers and the interrupt length), with an additional change to the `flank_threshold`.
* a configuration to use for `barcode02` from kit `Custom-Kit` with different primers and an interrupt length of 10, but with no change to the flank threshold.

### Configuration Options

| Option | Description |
| -- | -- |
| front_primer | Front primer sequence for cDNA |
| rear_primer | Rear primer sequence for cDNA |
| plasmid_front_flank | Front flanking sequence of poly(A) in plasmid |
| plasmid_rear_flank | Rear flanking sequence of poly(A) in plasmid |
| flank_threshold  | Threshold to use for detection of the flank/primer sequences. Equates to `(1 - edit distance / flank_sequence)` |
| primer_window | Window of bases at the front and rear of the rear within which to look for primer sequences |
| tail_interrupt_length | Combine tails that are within this distance of each other (default is 0, i.e. don't combine any) |
