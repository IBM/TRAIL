# Instructions for setting up the reasoner environment (Beagle)

To setup and run the reasoner environment you need to run the following commands:

Steps 1,3,4 and 5 are needed only once unless the code is changed.

1. Install the requirements:

    ``pip install -r requirements.txt``

2. These commands needs to run from the "code/reasoner_environment" directory

    ``cd code/reasoner_environment/``
3. Compile CloneableBeagle code 

    ``./compile_beagle.sh``
4. Generate gRPC stub code (necessary python code for the gRPC client)

    ``./generate_grpc_stub.sh``
    
5. Add ``code/reasoner_environment/beagle_grpc_stub/`` to your ``PYTHONPATH`` 

6. Start the reasoner environment:

    ``./start_environment.sh``
     
    By default only one instance of the reasoner will be launched on port 11111.
    1. ``$1`` is the number of instances
    2. ``$2`` is the port of the first instance
    
7. In order to use Beagle parse add these parameters to your run: ``--use_external_parser=1 --external_parser_port=11111``
