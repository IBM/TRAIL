#!/usr/bin/env bash
# cd stateless_beagle
# sbt assembly
SCRIPT_PATH=`dirname "$0"`

#number_of_environment=${1:-1}
port=${1:-11111}
#end_port=$((number_of_environment + start_port ))
nb_threads=${2:-24000}
xms=${3:-12}
xmx=${4:-190}
echo $port
# for (( port=$start_port; port<$end_port; port++ ))
# do
	echo "Starting Reasoner Server at localhost:$port"
	echo  -Xms${xms}g -Xmx${xmx}g  -Dscala.concurrent.context.minThreads=$nb_threads -Dscala.concurrent.context.maxThreads=$nb_threads -cp $SCRIPT_PATH/stateless_beagle/
	export GRPC_TRACE=all
	export GRPC_VERBOSITY=DEBUG
	java  -Xms${xms}g -Xmx${xmx}g  -Dscala.concurrent.context.minThreads=$nb_threads -Dscala.concurrent.context.maxThreads=$nb_threads -cp $SCRIPT_PATH/stateless_beagle/target/scala-2.11/stateless_beagle.jar  beagle.grpc.ProverServer $port
# done
