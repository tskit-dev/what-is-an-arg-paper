# what-is-an-arg-paper
Manuscript and code for the "A general and efficient representation of ARGS" paper

Create the paper by running `make` in the top level.

## Creating graphics

Install the necessary python libraries:

```
python -m pip install -r requirements.txt
```

Then to recreate plots used in the paper, type `make clean && make`

### Re-running inferences

To also re-run the software used for the inference plots, first install the
necessary inference tools by:

```
cd tools
make  # This will require python2 to be installed, to run ARGweaver conversion software
```

Version of these software tools and parameters used to create the plot
of inference results are all listed in [tools.config](./tools.config).

You can then force a re-run by removing everything in the `examples` directory, then
in the top level, repeating

```
make clean
make
```
