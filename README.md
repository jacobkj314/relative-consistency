# Relative Consistency


## Citation

coming soon


## Installation

Simply clone the repository:

```
git clone https://github.com/jacobkj314/relative-consistency
```

and then place the file relative_consistency.py where you need to use it

## Use

This measurement is intended for use with contrastive datasets such as [CondaQA](https://github.com/AbhilashaRavichander/CondaQA) or [ScoNe](https://github.com/selenashe/ScoNe/), from which you can construct pairs of related instances which vary in a feature, such as the presence, phrasing, or syntactic position of negation

After running 
```
from relative_consistency import relative_consistency
```