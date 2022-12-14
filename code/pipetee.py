# The intended use case is to copy data from one pipe to another, 
# whereby we also copy the input to a file.
# The idea is that if the writer of the input dies,
# then we know that the file is invalid and don't create it;
# if the reader of the output dies, then we continue to let the reader run and
# save the input to a file so it can be re-used in a subsequent run,
# and possibly to be used to debug the reader.
# The advantage of using pipes instead of just regular files to communicate
# is that the reader and writer can run concurrently, without complication;
# using files the reader would have to repeatedly sleep and check if more input is available.
    