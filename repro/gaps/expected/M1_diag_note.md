# M1 sanity failure: diagnosis (excerpt)

Verbatim from road-to-robotics `projects/jepa-surprise-receptive-field/README.md` lines 7-8 and 27-28 @ fb68283.

M1 (W1) failed the frozen sanity bar: class 5 step 2 AUROC 0.775 < 0.9 (`results/M1_table.md` l.60). Per PROTOCOL.md
and amendment O1 the M1 table is not read.

G0 reproduces with this scorer and checkpoint, so the load and the encoder/predictor path are not faulty and M1's 0.775
stands as measured. The M1 table stays unread under the frozen bar; nothing here re-reads or rescores it.
