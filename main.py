import argparse
import os

from src.h_CPP.Environment import H_CPPEnvironmentParams, H_CPPEnvironment
from src.CPP.Environment import CPPEnvironment, CPPEnvironmentParams
from src.DH.Environment import DHEnvironmentParams, DHEnvironment
from src.DHMulti.Environment import DHMultiEnvironment

from utils import *


def main_cpp(p):
    env = CPPEnvironment(p)

    env.run()


def main_dh(p):
    env = DHEnvironment(p)

    env.run()

def main_h(p):
    env = H_CPPEnvironment(p)

    env.run()


def main_dh_multi(p):
    env = DHMultiEnvironment(p)

    env.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Activates usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')

    parser.add_argument('--params', nargs='*', default=None)

    parser.add_argument('--h_cpp', action='store_true', help='Run Coverage Path Planning with hierarchical Agent')
    parser.add_argument('--cpp', action='store_true', help='Run Coverage Path Planning')
    parser.add_argument('--dh', action='store_true', help='Run Path Planning for Data Harvesting')
    parser.add_argument('--multi', action='store_true', help='Run Path Planning for Multi (So far only DH)')

    args = parser.parse_args()

    if args.generate_config:
        if args.h_cpp:
            generate_config(H_CPPEnvironmentParams(), "config/old_configs/h_cpp.json")
        elif args.dh:
            generate_config(DHEnvironmentParams(), "config/old_configs/dh.json")
        elif args.cpp:
            generate_config(CPPEnvironmentParams(), "config/old_configs/cpp.json")
        else:
            print("Specify which config to generate, DH or CPP")
        exit(0)

    if args.config is None:
        print("Config file needed!")
        exit(1)

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id

    save_json_to_logs(params, params.model_stats_params.save_model + f'{params.model_stats_params.log_file_name}/')
    save_json_to_logs(params, 'logs/plots/' + str(params.model_stats_params.log_file_name) + '/')


    # if args.cpp:
    #     main_cpp(params)
    # elif args.h_cpp:
    #     main_h(params)
    # elif args.dh:
    #     if args.multi:
    #         main_dh_multi(params)
    #     else:
    #         main_dh(params)
    main_h(params)