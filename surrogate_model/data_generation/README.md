# Teacher Data Generation

This branch generates paired reflection-coefficient targets
from the first JCAL output `Reflect`:

```matlab
R_real = real(Reflect);
R_imag = imag(Reflect);
```

Generate both arrays during the same teacher call and save them as `Y_re` and
`Y_im` with common `X`, `freq_grid`, and metadata. Do not use the existing JCAL
outputs named `Re` and `Im`; those are normalized surface-impedance components.

Inspection must validate shape, finiteness, frequency alignment, target ranges,
and numerical equality to the corresponding parts of `Reflect` on checked
samples. There is no component clipping.

The migration is implemented and was checked on a deterministic 50-curve
dataset; checked samples matched the complex `Reflect` values exactly.
