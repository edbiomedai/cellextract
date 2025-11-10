#!/usr/bin/env python
from argparse import ArgumentParser


class BaseParser:
    """Base class for parsers used in reading, writing and processing images and masks.
    This is a parent class that should be inherited by other parsers.

    Methods:
    `get_parser`: returns the parser object
    `get_args`: returns the parsed arguments
    `add_args`: adds arguments to the parser
    `remove_argument`: removes argument from the parser
    """

    def __init__(self):
        pass

    def get_parser(self):
        return self.parser

    def get_args(self):
        args = self.get_parser().parse_args()
        arg_dict = vars(args)
        if (
            "channel_names" in arg_dict
            and arg_dict["channel_names"] is None
            and "img" in arg_dict
        ):
            args.channel_names = [f"W{i+1}" for i in range(len(args.img))]
        if "mask_names" in arg_dict and args.mask_names is None and "mask" in arg_dict:
            if args.mask is None:
                args.mask_names = []
            else:
                args.mask_names = [f"mask{i+1}" for i in range(len(args.mask))]
        return args

    # Taken from https://stackoverflow.com/a/49753634
    def remove_argument(self, arg):
        for action in self.parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                self.parser._remove_action(action)
                break

        for action in self.parser._action_groups:
            for group_action in action._group_actions:
                if group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return

    def add_args(self, args, help, **kwargs):
        self.parser.add_argument(args, help=help, **kwargs)


def _add_granularity_args(bparser):
    bparser.add_args_to_both(
        "--subsample_size",
        nargs="+",
        help="Granularity - Subsampling factor for granularity measurements",
        required=False,
        default=0.25,
        type=float,
    )

    bparser.add_args_to_both(
        "--image_sample_size",
        nargs="+",
        help="Granularity - Subsampling factor for background reduction",
        required=False,
        default=0.25,
        type=float,
    )

    bparser.add_args_to_both(
        "--element_size",
        nargs="+",
        help="Granularity - Radius of structuring element",
        required=False,
        default=10,
        type=int,
    )

    bparser.add_args_to_both(
        "--granular_spectrum_length",
        nargs="+",
        help="Granularity - Range of the granular spectrum",
        required=False,
        default=16,
        type=int,
    )

    return bparser


class BaseSingleParser(BaseParser):
    """Base class for parsers used in reading, writing and processing images and masks.

    Constructor arguments:
    reqs: list of required arguments
            Options: "sample", "img_path", "mask_path", "out_path", "channel_names",
            "mask_names", "distance", or None (all)
    description: description of the parser
    exclude: list of arguments to exclude
    parser: if desired, existing parser object to add arguments to

    Methods:
    `get_parser`: returns the parser object
    `get_args`: returns the parsed arguments
    `add_args`: adds arguments to the parser
    `remove_argument`: removes argument from the parser
    """

    def __init__(self, reqs=None, description="Parser", exclude=None, parser=None):
        super().__init__()
        if reqs is None:
            reqs = [
                "sample",
                "img_path",
                "mask_path",
                "out_path",
                "channel_names",
                "mask_names",
                "distance",
            ]
        parser = ArgumentParser(description=description) if parser is None else parser

        if "sample" in reqs:
            parser.add_argument("--sample", type=str, help="Sample name", required=True)

        if "img_path" in reqs:
            parser.add_argument(
                "--img",
                nargs="+",
                help="Path to input image",
                required=True,
            )

        if "mask_path" in reqs:
            parser.add_argument(
                "--mask",
                action="append",
                nargs="+",
                help="Input label mask",
                required=False,
                default=None,
            )

        if "out_path" in reqs:
            parser.add_argument("--out", type=str, help="Output file", required=True)

        if "channel_names" in reqs:
            parser.add_argument(
                "--channel_names",
                nargs="+",
                help="Names of channels",
                required=False,
                default=None,
            )

        if "mask_names" in reqs:
            parser.add_argument(
                "--mask_names",
                nargs="+",
                help="Names of masks",
                required=False,
                default=None,
            )

        if "distance" in reqs:
            parser.add_argument(
                "--distance",
                type=int,
                default=1,
                help="Distance between pixels to be used in the calculation of the texture measurements",
            )

        self.parser = parser
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for arg in exclude:
                self.remove_argument(arg)


class BaseMultiParser(BaseParser):
    """
    Base class for parsers used in reading, writing and processing images and masks.
    This class specifically handles sample sheet-based processing.

    Constructor arguments:
    parser: if desired, existing parser object to add arguments to
    """

    def __init__(self, reqs=None, description=None, parser=None) -> None:
        super().__init__()
        parser = ArgumentParser(description=description) if parser is None else parser

        parser.add_argument(
            "--sample_sheet", type=str, help="Path to sample sheet", required=True
        )
        parser.add_argument(
            "--index",
            type=int,
            help="Index/indeces of sample to process (1-based)",
            required=True,
            default=None,
            nargs="+",
        )

        parser = BaseSingleParser(
            reqs=reqs,
            parser=parser,
            exclude=["img", "out", "sample", "mask"],
        ).get_parser()
        self.parser = parser


class BaseSuperParser(BaseParser):
    """
    Base class for superparsers used in reading, writing and processing images and masks.
    This is a class that instantiates a parser with two subparsers: single and multi.
    This allows to switch between processing one or multiple image sets.
    """

    def __init__(self, description=None, reqs_single=None, reqs_multi=None):
        super().__init__()
        parser = ArgumentParser(description=description)

        # Create subparsers
        subparsers = parser.add_subparsers(required=True, dest="mode")
        single = subparsers.add_parser("single", help="Process single image set")
        multi = subparsers.add_parser(
            "multi", help="Process multiple image sets using a sample sheet"
        )

        single = BaseSingleParser(parser=single, reqs=reqs_single)
        multi = BaseMultiParser(parser=multi, reqs=reqs_multi)

        self.single = single
        self.multi = multi
        self.parser = parser

    def get_single_parser(self):
        return self.single

    def get_multi_parser(self):
        return self.multi

    def add_args_to_both(self, arg, help="", **kwargs):
        self.single.add_args(arg, help=help, **kwargs)
        self.multi.add_args(arg, help=help, **kwargs)
