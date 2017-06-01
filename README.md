# ClassTM

Class(ifying)T(opic)M(odeling) is a framework for exploring supervised anchor
words, as detailed in *Is Your Anchor Going Up or Down?  Fast and Accurate
Supervised Topic Models*, Nguyen et al., NAACL 2015.  It is a counterpart to
[ActiveTM](https://github.com/nOkuda/activetm).  Whereas ActiveTM deals with
regression tasks, ClassTM deals with classification tasks.

## Prerequisites

The necessary Python libraries are listed in `requirements.txt`.  You can
install them via

```
pip install -r requirements.txt
```

in the `ClassTM` directory.

You will also need to compile the code in `classtm/ldac`.

You will also need to compile the code in `classtm/simplex`.

Finally, you will need to download SVMLight from
http://download.joachims.org/svm_light/current/svm_light.tar.gz, extract the
contents of the tarball in the `classtm/svm_light` directory, and compile the
code there.

## Settings

Two lines have been added to the settings file.

The first is 'anchors\_file  {name of anchors file}'. This is optional, and
only needs to be in the file if user-defined anchors are being used.

The second is 'lda\_helper  {variational, sampling}'. This defines whether LDA
will be done using variational or sampling methods.
