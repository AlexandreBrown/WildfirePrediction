from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource


class CanadaBoundary:
    def __init__(self, data_source: CanadaBoundaryDataSource) -> None:
        self.data_source = data_source
    
    def load(self, exclude_area_above_60_degree: bool = True):
        self.boundary = self.data_source.download()
        if exclude_area_above_60_degree:
            self.remove_area_above_60_degree()

    def remove_area_above_60_degree(self):
        self.boundary = self.boundary[(self.boundary['PRUID'] != '60') & (self.boundary['PRUID'] != '61') & (self.boundary['PRUID'] != '62')]