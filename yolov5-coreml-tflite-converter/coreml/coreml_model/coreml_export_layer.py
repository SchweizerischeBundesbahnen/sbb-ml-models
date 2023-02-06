import logging

import torch

from helpers.constants import XY_SLICE, WH_SLICE, SCORE_SLICE, CLASSES_SLICE, BLUE, END_COLOR, DETECTION, \
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, IOU_NAME, CONF_NAME, DEFAULT_MAX_NUMBER_DETECTION, MASKS_NAME, \
    CONFIDENCE_NAME, COORDINATES_NAME, NUMBER_NAME


class CoreMLExportLayerGenerator:
    """ Class that adds layers to a NeuralNetworkBuilder

    Attributes
    ----------
    model: ModelWrapper
        The model after which we want to add those export layers
    """

    def __init__(self, model):
        self.model = model

    def add_to(self, builder):
        """ Creates export logic and adds it to the builder

         Parameters
         ----------
         builder: NeuralNetworkBuilder
            The CoreML neural network builder in which we add the layers
         """
        if self.model.model_type == DETECTION:
            self.__add_export_detection(builder)
        else:
            self.__add_export_segmentation(builder)

    def __add_export_detection(self, builder):
        # Adds the yolov5 export layer to the coreml model for detection
        logging.info(f"{BLUE}Adding export layers...{END_COLOR}")
        outputNames = [output.name for output in builder.spec.description.output]

        nb_anchors = []
        # (1, nA, nC, nC, nO), (1, nA, nC, nC, nO), (1, nA, nC, nC, nO)
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
                                    alpha=self.model.strides[i].item())

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
            (nb_predictions, self.model.number_of_classes), (nb_predictions, 4)])

    def __add_export_segmentation(self, builder):
        # Adds nms and masks calculations for Yolov5 segmentation
        logging.info(f"{BLUE}Adding export layers...{END_COLOR}")

        masks_resolution = self.model.input_resolution // 4

        ## Coordinates calculation ##
        # input[0:4]
        builder.add_slice(name=f"slice_coordinates", input_name="all_predictions",
                          output_name=f"sliced_coordinates", axis="width", start_index=XY_SLICE[0],
                          end_index=WH_SLICE[1])
        # Normalize
        builder.add_scale(name=f"normalize_coordinates", input_name=f"sliced_coordinates",
                          output_name=f"temp_raw_coordinates",
                          W=torch.tensor([1 / self.model.input_resolution]).numpy(), b=0, has_bias=False)
        ### Confidence calculation ###
        # input[4:5] = object confidence
        builder.add_slice(name=f"slice_object_confidence", input_name="all_predictions",
                          output_name=f"object_confidence", axis="width", start_index=SCORE_SLICE[0],
                          end_index=SCORE_SLICE[1])
        # input[5:nb_classes+5] = label confidence
        builder.add_slice(name=f"slice_label_confidence", input_name="all_predictions",
                          output_name=f"label_confidence", axis="width", start_index=CLASSES_SLICE[0],
                          end_index=CLASSES_SLICE[0] + self.model.number_of_classes)
        # confidence = object_confidence * label_confidence
        builder.add_multiply_broadcastable(name=f"multiply_object_label_confidence", input_names=[
            f"label_confidence", f"object_confidence"],
                                           output_name=f"temp_raw_confidence")

        builder.add_nms(name=f"nms", input_names=["temp_raw_coordinates", "temp_raw_confidence", IOU_NAME, CONF_NAME],
                        output_names=["nms_coordinates", "nms_confidence", "nms_indices", NUMBER_NAME],
                        iou_threshold=DEFAULT_IOU_THRESHOLD, score_threshold=DEFAULT_CONF_THRESHOLD,
                        max_boxes=DEFAULT_MAX_NUMBER_DETECTION)

        builder.add_flatten_to_2d(
            name=f"flatten_coordinates", input_name="nms_coordinates",
            output_name=COORDINATES_NAME, axis=-1)

        builder.add_flatten_to_2d(
            name=f"flatten_confidence", input_name="nms_confidence",
            output_name=CONFIDENCE_NAME, axis=-1)

        ### Masks calculations ###
        builder.add_slice(name=f"slice_masks", input_name="all_predictions",
                          output_name=f"sliced_masks", axis="width",
                          start_index=CLASSES_SLICE[0] + self.model.number_of_classes,
                          end_index=0)

        # Gather mask
        builder.add_gather(name="gather_masks", input_names=["sliced_masks", "nms_indices"],
                           output_name="gathered_masks", axis=1)

        # (nbDet, 32)
        builder.add_flatten_to_2d(
            name=f"flatten_masks", input_name="gathered_masks",
            output_name=f"flattened_masks", axis=-1)

        # (1, 32, 160, 160) -> (1, 32, 160*160)
        builder.add_reshape_static(name="reshape_protos", input_name="segmentation_protos",
                                   output_name="reshaped_protos", output_shape=(1, 32, masks_resolution ** 2))

        # (1, 32, 160*160) -> (32, 160*160)
        builder.add_flatten_to_2d(
            name=f"flatten_protos", input_name="reshaped_protos",
            output_name=f"flattened_protos", axis=-1)

        # (nbDet, 32) . (32, 160*160) = (nbDet, 160*160)
        builder.add_batched_mat_mul(name="matmul_protos", input_names=["flattened_masks", "flattened_protos"],
                                    output_name="matrix_protos")

        # (nbDet, 160, 160)
        builder.add_reshape_static(name="reshaped_matmul_protos", input_name="matrix_protos", output_name="temp_masks",
                                   output_shape=(-1, masks_resolution, masks_resolution))

        # Scale coordinates to mask
        builder.add_elementwise(name=f"scale_coordinates_to_masks", input_names=[
            COORDINATES_NAME], output_name=f"scaled_coordinates",
                                mode="MULTIPLY", alpha=masks_resolution)

        # XYWH to XYXY (1, nbDet, 1) for each
        builder.add_slice(name="slice_x", input_name="scaled_coordinates", output_name="sliced_x", axis="width",
                          start_index=0, end_index=1)
        builder.add_slice(name="slice_y", input_name="scaled_coordinates", output_name="sliced_y", axis="width",
                          start_index=1, end_index=2)
        builder.add_slice(name="slice_w", input_name="scaled_coordinates", output_name="sliced_w", axis="width",
                          start_index=2, end_index=3)
        builder.add_slice(name="slice_h", input_name="scaled_coordinates", output_name="sliced_h", axis="width",
                          start_index=3, end_index=4)

        builder.add_elementwise(name="divide_h_by_2", input_names=["sliced_h"], output_name="h_divided_by_2",
                                mode="MULTIPLY", alpha=1 / 2)
        builder.add_elementwise(name="divide_w_by_2", input_names=["sliced_w"], output_name="w_divided_by_2",
                                mode="MULTIPLY", alpha=1 / 2)
        builder.add_elementwise(name="multiply_h_divided_by_2_by_minus_1", input_names=["h_divided_by_2"],
                                output_name="h_divided_by_2_minus",
                                mode="MULTIPLY", alpha=-1)
        builder.add_elementwise(name="multiply_w_divided_by_2_by_minus_1", input_names=["w_divided_by_2"],
                                output_name="w_divided_by_2_minus",
                                mode="MULTIPLY", alpha=-1)

        builder.add_elementwise(name=f"compute_x1", input_names=[
            "sliced_x", "w_divided_by_2_minus"], output_name=f"x1_temp", mode="ADD")
        builder.add_elementwise(name=f"compute_y1", input_names=[
            "sliced_y", "h_divided_by_2_minus"], output_name=f"y1_temp", mode="ADD")
        builder.add_elementwise(name=f"compute_x2", input_names=[
            "sliced_x", "w_divided_by_2"], output_name=f"x2_temp", mode="ADD")
        builder.add_elementwise(name=f"compute_y2", input_names=[
            "sliced_y", "h_divided_by_2"], output_name=f"y2_temp", mode="ADD")
        builder.add_reshape_static(name="reshape_x1", input_name="x1_temp", output_name="x1", output_shape=(-1, 1, 1))
        builder.add_reshape_static(name="reshape_y1", input_name="y1_temp", output_name="y1", output_shape=(-1, 1, 1))
        builder.add_reshape_static(name="reshape_x2", input_name="x2_temp", output_name="x2", output_shape=(-1, 1, 1))
        builder.add_reshape_static(name="reshape_y2", input_name="y2_temp", output_name="y2", output_shape=(-1, 1, 1))

        # masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
        r = torch.arange(masks_resolution, dtype=torch.float).view(1, 1, masks_resolution)  # rows shape(1,w,1)
        c = torch.arange(masks_resolution, dtype=torch.float).view(1, masks_resolution, 1)  # cols shape(h,1,1)
        builder.add_load_constant(name="load_constant_r", output_name="constant_r", constant_value=r,
                                  shape=(1, masks_resolution, 1))
        builder.add_load_constant(name="load_constant_c", output_name="constant_c", constant_value=c,
                                  shape=(masks_resolution, 1, 1))
        builder.add_reshape_static(name="reshape_r", input_name="constant_r", output_name="reshaped_r",
                                   output_shape=(1, 1, masks_resolution))
        builder.add_reshape_static(name="reshape_c", input_name="constant_c", output_name="reshaped_c",
                                   output_shape=(1, masks_resolution, 1))

        builder.add_greater_than(name="compute_r_greater_or_equal_than_x1", input_names=["reshaped_r", "x1"],
                                 output_name="r_greater_or_equal_than_x1", use_greater_than_equal=True)
        builder.add_greater_than(name="compute_x2_greater_than_r", input_names=["x2", "reshaped_r"],
                                 output_name="x2_greater_than_r", use_greater_than_equal=False)
        builder.add_elementwise(name="multiply_masks_1",
                                input_names=["r_greater_or_equal_than_x1", "x2_greater_than_r"],
                                output_name="crop_masks1", mode="MULTIPLY")

        builder.add_greater_than(name="compute_c_greater_or_equal_to_y1", input_names=["reshaped_c", "y1"],
                                 output_name="c_greater_or_equal_to_y1", use_greater_than_equal=True)
        builder.add_greater_than(name="compute_y2_greater_than_c", input_names=["y2", "reshaped_c"],
                                 output_name="y2_greater_than_c", use_greater_than_equal=False)
        builder.add_elementwise(name="multiply_masks_2",
                                input_names=["c_greater_or_equal_to_y1", "y2_greater_than_c"],
                                output_name="crop_masks2", mode="MULTIPLY")
        builder.add_elementwise(name="multiply_crop_masks", input_names=["crop_masks1", "crop_masks2"],
                                output_name="crop_masks", mode="MULTIPLY")

        builder.add_elementwise(name="multiply_final_masks", input_names=["temp_masks", "crop_masks"],
                                output_name="final_masks", mode="MULTIPLY")

        builder.add_resize_bilinear(name="resize_masks", input_name="final_masks", output_name="resized_masks",
                                    target_height=self.model.input_resolution, target_width=self.model.input_resolution,
                                    mode="UPSAMPLE_MODE")

        builder.add_clip(name="clip_masks", input_name="resized_masks", output_name=MASKS_NAME,
                         min_value=0.5, max_value=1.0)

        builder.set_output(output_names=[COORDINATES_NAME, CONFIDENCE_NAME, MASKS_NAME, NUMBER_NAME], output_dims=[
            (DEFAULT_MAX_NUMBER_DETECTION, self.model.number_of_classes), (DEFAULT_MAX_NUMBER_DETECTION, 4),
            (DEFAULT_MAX_NUMBER_DETECTION, masks_resolution, masks_resolution), (1,)])


def make_grid(anchors, stride, nx=20, ny=20, i=0):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).view((ny, nx, 2)).float()
    anchor_grid = (anchors[i].clone() * stride[i]).view((1, len(anchors[i]), 1, 1, 2)).expand(
        (1, len(anchors[i]), ny, nx, 2)).float()
    return grid.numpy(), anchor_grid.numpy()
