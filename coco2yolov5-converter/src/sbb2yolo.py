""" Main """
import argparse
import kwcoco
import logging
import os
import shutil
import shutil
import sys
from pathlib import Path
from random import random

from utils.constants import *
from utils.coordinates_util import convertRelCoco2Yolo, convertAbsCoco2Yolo
from utils.txt_util import TxtUtil
from utils.yaml_util import YamlUtil
from utils.yolo_dataset import YoloDataset
from utils.coco_datasets import CocoDatasets


# TODO: Add param
# valid_tag_names = ["Abfahrtsmonitor","Abfallbehälter-Abteil","Abfallbehälter-Abteil_fehlt","Abfallbehälter-Bahnhof","Abfallbehälter-Vorraum","Abfallbehälter_Graffiti","Abfalleimer-Bahnhof","Abfalleimer-Bahnhof_Graffiti","Abteil-Übergangstüre","Abteilwand_Graffiti","Analoge-Schliessfächer","Analoge-Schliessfächer_Verkratzt","Anderes-Piktogramm-Bahnhof","Ankunftsmonitor","Armlehne","Auto","Automatennummer","Bedien-und-Anzeigeelemente","Beleuchtung im Fst.","Beleuchtung-Bahnhof_AUS-defekt","Beleuchtung-Bahnhof_EIN","Beleuchtung-Wagen_AUS-defekt","Beleuchtung-Wagen_EIN","Betriebslagemonitor","Billettautomat-SBB","Billettautomat-SBB_Bildschirm-eingeschlagen","Billettautomat-SBB_Graffiti","Billettentwerter-SBB","Billettentwerter_Graffiti","Blaue-FIS-Tafel","Busnummer","CAB-Radio und Funkgerät","Dienstbeleuchtung-Schienenfz","Digitale-Schliessfächer","DMI_Bedien_Diagnosebildschirm","Einstiegsbereich-Wand_Graffiti","Einstiegstüre","Einstiegstüre_offen","ETCS","Fenster-innen-Wagen","Fenster-innen-Wagen_Graffiti","Feuerlöscher","FIS_Graffiti","Fluchthaube","Formulare","Fussboden-Bahnhof_beschädigt","Fussboden-Bahnhof_verschmutzt","Fussboden-Wagen_Verschmutzt","Fusspodest-Pedal","Füllstandsanz-Bioreaktor-Fäkalientank","Füllstandsanz-Wasser","Gebäudetüre","Gebäudetüre_Graffiti","Geldeinwurf","Generalanzeiger","Gleis-Piktogramm-Tafel","Handlauf-Bahnhof","Inventar Fzg.","Kamera-Wagen","KIS oder TIMS","KIS-Anzeige-aussen","KIS-Anzeige-innen","Klima und Heizung","Kupplung","Lautsprecher Fst.","Lautsprecher-Bahnhof","LEA-Halter und USB","Lift","Monitor-Wagen","Motorrad","Nothammer","Nothammer_fehlt","Pantograf","Parkautomat","Perronanzeiger","Perronsäule_Graffiti","Person","Pflanzenwuchs","Puffer-Stossvorrichtung","Quittierschalter","Re 460","Recyclingstation","Rolltreppe","Rollvorhang","Rückspiegel-und-Rücksehsysteme","Sander","SBB-Uhr","Scheibenwisch-und-Waschanlage","Schienenfz","Schienenfz_Graffiti","Schienenfz_Verschmutzung","Schlauch-Kupplung","Schnee","Seitenfenster Fst.","Sektor-Piktogramm-Tafel","Selecta-Kaffeeautomat","Selectaautomat","Selectaautomat_eingeschlagen","Sitz Lokführer","Sitz-Wagen","Sitz-Wagen_Kopfschutztuch-fehlt","Sitz-Wagen_Polster-aufgerissen","Sitz-Wagen_Polster-fehlt","Sitz-Wagen_Polster-verschmutzt","Sitzbank-Bahnhof","Sitzbank-Bahnhof_Graffiti","Sitzbank-Bahnhof_Verschmutzt","Sitzverstellung-Wagen","Sonnenschutz-Führerstand","Sonstige-Wagen-Graffitis","Sprechstelle-Zugpersonal","Steckdose-Wagen","Sticker","Text-auf-FIS-Tafel","Tisch-Wagen","Tisch-Wagen_Graffiti","Tisch-Wagen_Verschmutzt","Tor","Treppe-Bahnhof","Treppe-Bahnhof_Verschmutzt","Türe-Führerstand","Türgriff_Einstieg","Türknopf_Einstieg_schliessen","Türknopf_Einstieg_öffnen","UIC-Nummer","Umgebung_Graffiti","Unter-Überführung-Rampe_Graffiti","Velo","Wartesaal-Türe","WC-Bahnhof_Graffiti","WC-Kabine-Abfallbehälter","WC-Kabine-Defekt-Kleber","WC-Kabine-Oberfläche-Waschbereich-verkratzt","WC-Kabine-Seifenspender","WC-Kabine-Spiegel_Graffiti","WC-Kabine-Toilette","WC-Kabine-Türe","WC-Kabine-Türklinke-Griff","WC-Kabine-Türknopf","WC-Kabine-Wand_Graffiti","WC-Kabine-Waschbecken","WC-Kabine-Wasserhahn","WC-Türe-Bahnhof","ZUB"]

class Coco2YoloDatasetConverter:
    def __init__(self, coco_input_folders: list, yolo_output_folder: str, yolo_config_file: str,
                 dataset_split_pivot: float, new_format: bool = True, overwrite: bool = False):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.coco_input_folders = coco_input_folders
        self.yolo_output_folder = yolo_output_folder.rstrip('/')
        self.yolo_config_file = yolo_config_file
        self.dataset_split_pivot = dataset_split_pivot
        self.new_format = new_format
        self.overwrite = overwrite

    def convert(self):
        """ Convert Coco Annotations from JSON file into YoloV5 Structure """
        yolo_dataset = YoloDataset(self.yolo_output_folder, self.overwrite, self.dataset_split_pivot)
        logging.info("\033[36mStarting conversion from Coco to Yolo format...\033[0m")

        labels, cocos, image_folders = CocoDatasets(self.coco_input_folders).read_datasets()

        # TODO: Add param
        # annotation_names_all = [a for a in annotation_names_all if a in valid_tag_names]

        # Convert all input coco datasets
        for i, (coco, images_basedir) in enumerate(zip(cocos, image_folders)):
            logging.info(f"Converting dataset ({i + 1}/{len(self.coco_input_folders)})")
            yolo_dataset.create(coco, images_basedir, labels)

        # Write yaml config file
        yolo_dataset.writeYaml(labels, self.yolo_config_file, self.new_format)

        logging.info(f"\033[32mConversion success\033[0m: saved in {self.yolo_output_folder}")

        logging.info(
            f"- The dataset contains {len(list(Path(yolo_dataset.image_train).glob('**/*')))} images in train and {len(list(Path(yolo_dataset.image_val).glob('**/*')))} images in val")
        logging.info(f"- There are {len(labels)} labels")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-input-folders', dest='coco_input_folders', type=str, nargs='+',
                        help='Paths to COCO dataset', required=True)
    parser.add_argument('--yolo-output-folder', dest='yolo_output_folder', type=str, default='output',
                        help='Output folder (where to save Yolo dataset). Default: ./output')
    parser.add_argument('--yolo-config-file', dest='yolo_config_file', type=str,
                        default='config.yaml', help='YoloV5 YAML file name. Default: config.yaml')
    parser.add_argument('--dataset-split-pivot', dest='dataset_split_pivot', type=float, default=0.8,
                        help='Train/val split (0.8 = 80 percent in train, 20 percent in val). Default: 0.8')
    parser.add_argument('--new-format', action='store_true', help="If set, will use the slightly different 'new' format for yolo yaml config.")
    parser.add_argument('--overwrite', action='store_true',
                        help="If set, will overwrite output dataset if it already exist.")
    # TODO 7: add argument include labels
    # TODO 8: add argument exclude labels
    # TODO 9: add argument for move and copy
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    Coco2YoloDatasetConverter(args.coco_input_folders, args.yolo_output_folder,
                              args.yolo_config_file, args.dataset_split_pivot, args.new_format, args.overwrite).convert()
