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

To also re-run the software used for the inference plots, the
necessary inference tools need to be installed by running the makefile
in the tools directory (note that this
requires `python2` to be installed, to run ARGweaver conversion software)

```
make -C tools
```

Version of these software tools and parameters used to create the plot
of inference results are all listed in [tools.config](./tools.config).

You can then force a re-run by removing everything in the `examples` directory, then
recreating them by making the example inputs and outputs, followed by
repeating the graphics creating commands in the top level directory:

```
rm -rf examples/*
make clean
make inference_outputs
make
```
