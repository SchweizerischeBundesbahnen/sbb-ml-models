import logging

import torch

from helpers.constants import XY_SLICE, WH_SLICE, SCORE_SLICE, CLASSES_SLICE, BLUE, END_COLOR


class CoreMLExportLayerGenerator:
    def __init__(self, model):
        self.model = model

    def add_to(self, builder):
        '''
        Adds the yolov5 export layer to the coreml model
        '''
        logging.info(f"{BLUE}Adding export layers...{END_COLOR}")
        outputNames = [output.name for output in builder.spec.description.output]

        nb_anchors = []
        # (1, nA, nC, nC, nO), ...
        for i, outputName in enumerate(outputNames):
            grid, anchor_grid = make_grid(self.model.anchors, self.model.strides,
                                          self.model.feature_map_dimensions[i], self.model.feature_map_dimensions[i], i)

            # formulas: https://github.com/ultralytics/yolov5/issues/471
            builder.add_activation(name=f"sigmoid_{outputName}", non_linearity="SIGMOID",
                                   input_name=outputName, output_name=f"{outputName}_sigmoid")

            ### Coordinates calculation ###
            # input (1, nA, nC, nC, nO), output (1, nA, nC, nC, 2) -> nC = 640 / strides[i]
            # nA = number of anchors, nO = number of outputs (#classes + 5)
            # input[0:2] = x,y
            builder.add_slice(name=f"slice_coordinates_xy_{outputName}", input_name=f"{outputName}_sigmoid",
                              output_name=f"{outputName}_sliced_coordinates_xy", axis="width", start_index=XY_SLICE[0],
                              end_index=XY_SLICE[1])
            # x,y * 2
            builder.add_elementwise(name=f"multiply_xy_by_two_{outputName}", input_names=[
                f"{outputName}_sliced_coordinates_xy"], output_name=f"{outputName}_multiplied_xy_by_two",
                                    mode="MULTIPLY", alpha=2)
            # x,y * 2 - 0.5
            builder.add_elementwise(name=f"subtract_0_5_from_xy_{outputName}", input_names=[
                f"{outputName}_multiplied_xy_by_two"], output_name=f"{outputName}_subtracted_0_5_from_xy", mode="ADD",
                                    alpha=-0.5)
            # x,y * 2 - 0.5 + grid[i]
            # (1, nA, nC, nC, 2) + (nC, nC, 2)
            builder.add_bias(name=f"add_grid_from_xy_{outputName}", input_name=f"{outputName}_subtracted_0_5_from_xy",
                             output_name=f"{outputName}_added_grid_xy", b=grid, shape_bias=grid.shape)

            # (x,y * 2 - 0.5 + grid[i]) * stride[i]
            builder.add_elementwise(name=f"multiply_xy_by_stride_{outputName}", input_names=[
                f"{outputName}_added_grid_xy"], output_name=f"{outputName}_calculated_xy", mode="MULTIPLY",
                                    alpha=self.model.strides[i])

            # input (1, nA, nC, nC, nO), output (1, nA, nC, nC, 2)
            # input[2:4] = w,h
            builder.add_slice(name=f"slice_coordinates_wh_{outputName}", input_name=f"{outputName}_sigmoid",
                              output_name=f"{outputName}_sliced_coordinates_wh", axis="width", start_index=WH_SLICE[0],
                              end_index=WH_SLICE[1])
            # w,h * 2
            builder.add_elementwise(name=f"multiply_wh_by_two_{outputName}", input_names=[
                f"{outputName}_sliced_coordinates_wh"], output_name=f"{outputName}_multiplied_wh_by_two",
                                    mode="MULTIPLY", alpha=2)
            # (w,h * 2) ** 2
            builder.add_unary(name=f"power_wh_{outputName}", input_name=f"{outputName}_multiplied_wh_by_two",
                              output_name=f"{outputName}_power_wh", mode="power", alpha=2)

            # (w,h * 2) ** 2 * anchor_grid[i]
            # (1, nA, nC, nC, 2) * (1, nA, nC, nC, 2)
            builder.add_load_constant_nd(
                name=f"anchors_{outputName}", output_name=f"{outputName}_anchors", constant_value=anchor_grid,
                shape=anchor_grid.shape)
            builder.add_elementwise(name=f"multiply_wh_with_anchors_{outputName}", input_names=[
                f"{outputName}_power_wh", f"{outputName}_anchors"], output_name=f"{outputName}_calculated_wh",
                                    mode="MULTIPLY")

            # (x, y, w, h)
            builder.add_concat_nd(name=f"concat_coordinates_{outputName}", input_names=[
                f"{outputName}_calculated_xy", f"{outputName}_calculated_wh"],
                                  output_name=f"{outputName}_raw_coordinates", axis=-1)
            # Normalize coordinates
            builder.add_scale(name=f"normalize_coordinates_{outputName}", input_name=f"{outputName}_raw_coordinates",
                              output_name=f"{outputName}_raw_normalized_coordinates",
                              W=torch.tensor([1 / self.model.input_resolution]).numpy(), b=0, has_bias=False)

            ### Confidence calculation ###
            # input[4:5] = object confidence
            builder.add_slice(name=f"slice_object_confidence_{outputName}", input_name=f"{outputName}_sigmoid",
                              output_name=f"{outputName}_object_confidence", axis="width", start_index=SCORE_SLICE[0],
                              end_index=SCORE_SLICE[1])
            # input[5:] = label confidence
            builder.add_slice(name=f"slice_label_confidence_{outputName}", input_name=f"{outputName}_sigmoid",
                              output_name=f"{outputName}_label_confidence", axis="width", start_index=CLASSES_SLICE[0],
                              end_index=CLASSES_SLICE[1])
            # confidence = object_confidence * label_confidence
            builder.add_multiply_broadcastable(name=f"multiply_object_label_confidence_{outputName}", input_names=[
                f"{outputName}_label_confidence", f"{outputName}_object_confidence"],
                                               output_name=f"{outputName}_raw_confidence")

            # input: (1, 3, nC, nC, 85), output: (3 * nc^2, 85)
            builder.add_flatten_to_2d(
                name=f"flatten_confidence_{outputName}", input_name=f"{outputName}_raw_confidence",
                output_name=f"{outputName}_flatten_raw_confidence", axis=-1)
            builder.add_flatten_to_2d(
                name=f"flatten_coordinates_{outputName}", input_name=f"{outputName}_raw_normalized_coordinates",
                output_name=f"{outputName}_flatten_raw_coordinates", axis=-1)

            nb_anchors.append(len(self.model.anchors[i]))

        builder.add_concat_nd(name="concat_confidence", input_names=[
            f"{outputName}_flatten_raw_confidence" for outputName in outputNames], output_name="raw_confidence",
                              axis=-2)
        builder.add_concat_nd(name="concat_coordinates", input_names=[
            f"{outputName}_flatten_raw_coordinates" for outputName in outputNames], output_name="raw_coordinates",
                              axis=-2)

        nb_predictions = sum([nb_anchors[i] * (x ** 2) for i, x in enumerate(self.model.feature_map_dimensions)])
        builder.set_output(output_names=["raw_confidence", "raw_coordinates"], output_dims=[
            (nb_predictions, self.model.number_of_classes()), (nb_predictions, 4)])


def make_grid(anchors, stride, nx=20, ny=20, i=0):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).view((ny, nx, 2)).float()
    anchor_grid = (anchors[i].clone() * stride[i]).view((1, len(anchors[i]), 1, 1, 2)).expand(
        (1, len(anchors[i]), ny, nx, 2)).float()
    return grid.numpy(), anchor_grid.numpy()
