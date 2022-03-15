from tflite_support import metadata_schema_py_generated as _metadata_fb


class MetadataHelper:
    def _add_content_image(self, meta):
        # Image RGB
        meta.content = _metadata_fb.ContentT()
        meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        meta.content.contentProperties.colorSpace = (
            _metadata_fb.ColorSpaceType.RGB)
        meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.ImageProperties)

    def _add_content_feature(self, meta):
        meta.content = _metadata_fb.ContentT()
        meta.content.content_properties = (_metadata_fb.FeaturePropertiesT())
        meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.FeatureProperties)

    def _add_content_bounding_box(self, meta):
        # Bounding box y1, x1, y2, x2
        meta.content = _metadata_fb.ContentT()
        meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.BoundingBoxProperties)
        meta.content.contentProperties = (_metadata_fb.BoundingBoxPropertiesT())
        meta.content.contentProperties.index = [1, 0, 3, 2]
        meta.content.contentProperties.type = (_metadata_fb.BoundingBoxType.BOUNDARIES)
        meta.content.contentProperties.coordinateType = (_metadata_fb.CoordinateType.RATIO)

    def _add_normalization(self, meta, mean, std):
        normalization_unit = _metadata_fb.ProcessUnitT()
        normalization_unit.optionsType = (
            _metadata_fb.ProcessUnitOptions.NormalizationOptions)
        normalization_unit.options = _metadata_fb.NormalizationOptionsT()
        normalization_unit.options.mean = [mean]
        normalization_unit.options.std = [std]
        meta.processUnits = [normalization_unit]

    def _add_range(self, meta):
        meta.content.range = _metadata_fb.ValueRangeT()
        meta.content.range.min = 2
        meta.content.range.max = 2

    def _add_stats(self, meta, max, min):
        stats = _metadata_fb.StatsT()
        stats.max = [max]
        stats.min = [min]
        meta.stats = stats
