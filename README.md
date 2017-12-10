# Tisk 1.x Distribution

This code is to run a TISK model on Python 3.x.

# Files
There are three files in TISK 1.x.

Basic_TISK_Class.py<br>
Phoneme_Data.txt<br>
Pronunciation.txt<br>

# Pre-installed

This code require 'numpy', 'matplotlib' modules.
```
pip install numpy<br>
pip install matplotlab<br>
```

# TISK 1.0 code example
## Running TISK
### Preliminary steps
The following commands prepare TISK for simulations. Lines preceded by "#" are comments and can be skipped.
```
# load the TISK functions
import Basic_TISK_Class as tisk

# load the phoneme and pronunciation [word] lists and 
#prepare appropriate connections
phoneme_List, pronunciation_List = tisk.List_Generate()

# initialize the model with the the phoneme_List, 
# pronunciation_List, number of time slots, and threshold
tisk_Model = tisk.TISK_Model(phoneme_List, pronunciation_List,
                             time_Slots = 10,
                             nPhone_Threshold = 0.91)
```
### Initialize and / or modify parameters

Before running simulations, you must initialize the parameters. To use the default parameters of TISK 1.0 (Hannagan et al., 2013), just enter: 
```
# initialize the model with default or current parameters
tisk_Model.Weight_Initialize()
```

The model does not automatically initialize because this is the step where all connections are made, etc., and initialization can take a long time for a large model with thousands of words. To control specific categories of parameters or specific parameters, use the following examples: 
```
# change selected TISK parameters
tisk_Model.Decay_Parameter_Assign(
                    decay_Phoneme = 0.001,
                    decay_Diphone = 0.001,
                    decay_SPhone = 0.001,
                    decay_Word = 0.01)
tisk_Model.Weight_Parameter_Assign(
                    input_to_Phoneme_Weight = 1.0,
                    phoneme_to_Phone_Weight = 0.1,
                    diphone_to_Word_Weight = 0.05,
                    sPhone_to_Word_Weight = 0.01,
                    word_to_Word_Weight = -0.005)
tisk_Model.Feedback_Parameter_Assign(
                    word_to_Diphone_Activation = 0,
                    word_to_SPhone_Activation = 0,
                    word_to_Diphone_Inhibition = 0,
                    word_to_SPhone_Inhibition = 0)
tisk_Model.Weight_Initialize()
```

To modify a subset of parameters, just specify the subset, and the others will retain their current values. For example: 

```
tisk_Model.Decay_Parameter_Assign(
                    decay_Phoneme = 0.002,
                    decay_Diphone = 0.0005)
```

To list the current parameters, enter the following command:

```
tisk_Model.Parameter_Display()
```

## Simulate processing of a phoneme string and graph results for phonemes and words
Here is an example of a basic command that calls a simulation of the word 'pat' (technically, it is more correct to say "a simulation of the pronunciation 'pat'", since the user can specify pronunciations that are not in the lexicon [i.e., the pronunciation_List]):

```
# trigger a simulation without producing output;
# this prepares a model for inspection

tisk_Model.Display_Graph(pronunciation='pat')
```

On its own, this command doesn't do anything apparent to the user (though the simulation is in fact conducted). To create a graph that is displayed within an IDE, add arguments to display specific phonemes:

```
# trigger a simulation and create a phoneme input graph
tisk_Model.Display_Graph(
           pronunciation='pat',
           display_Phoneme_List = [('p', 0), ('a', 1), ('t', 2)])
```

This code means "input the pronunciation /pat/, and export a graph with phoneme activations for /p/, /a/, and /t/ in the first, second, and third positions, respectively".

We can extend this to create activation graphs of diphones, single phones, and words. The following example creates one of each:

```
# trigger a simulation and make 3 graphs
tisk_Model.Display_Graph(pronunciation='pat',
                         display_Diphone_List = ['pa', 'pt', 'ap'],
                         display_Single_Phone_List = ['p', 'a', 't'],
                         display_Word_List = ['pat', 'tap'])
```

To export the graphs in a standard graphics format (PNG), simply add one more argument to the command:
 
```
# trigger a simulation, make 3 graphs, save them as PNG files
tisk_Model.Display_Graph(pronunciation='pat',
                         display_Diphone_List = ['pa', 'pt', 'ap'],
                         display_Single_Phone_List = ['p', 'a', 't'],
                         display_Word_List = ['pat', 'tap'],
                         file_Save = True)
```

## Extract simulation data to a numpy matrix

The basic method for extracting data as a numpy matrix is as follows:
```
# trigger a simulation and ready data structures
# without creating any output
tisk_Model.Extract_Data(pronunciation='pat')
```

The method is similar to the graphing functions. The code above is the core command that readies appropriate structures for extraction, but it does not by itself generate any result for the user. To extract data, arguments specifying the details desired are required. For example, to get data corresponding to the word plot above (showing the activations of /pat/ and /tap/ given the input /pat/), the following command would put the data in a numpy matrix called 'result':

```
# trigger a simulation and assign data structures to 'result'
result = tisk_Model.Extract_Data(pronunciation='pat',
                                 extract_Word_List = ['pat', 'tap'])
```

When this command is executed, the 'result' variable becomes a list with length 1, consisting of a single numpy matrix with shape (2, 100). The first and second rows are the activation patterns of the word units for /pat/ and /tap/, respectively.

```
# trigger a simulation and assign data structures to 'result'

result = tisk_Model.Extract_Data(pronunciation='pat',
            extract_Phoneme_List = [('p', 0), ('a', 1), ('t', 2)],
            extract_Single_Phone_List = ['p', 'a', 't'])
```

Here, result becomes a list with length 2. The first item is a numpy matrix with the input unit activations for the 3 specified phonemes across the 100 steps of the simulation. The second is a numpy matrix with the activations of the specified single phonemes in the n-phone layer over the 100 steps of the simulation.



## Export simulation data to text files
## Batch simulation of multiple words
## Extract data for multiple words in text files
## Getting comprehensive data for every word in the lexicon
## Batch size control
## Reaction time and accuracy for specific words
## More complex simulations

# Reporting Issues

If you suspect that a bug has caused a malfunction while using the program, please report it using the issues feature of this repository.

https://github.com/CODEJIN/Tisk/issues

However, the current status of this project is occasional management. Therefore, it is difficult to respond promptly to reported issues. Thank you for your understanding.
