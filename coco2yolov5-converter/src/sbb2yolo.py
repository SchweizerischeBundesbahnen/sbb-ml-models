""" Main """
import argparse
import kwcoco
import shutil
import sys
from pathlib import Path
from random import random
import logging

from converter.convert_util import convertRelCoco2Yolo, convertAbsCoco2Yolo
from converter.filesystem_util import FileSystemUtil
from converter.txt_util import TxtUtil
from converter.yaml_util import YamlUtil
import shutil

# TODO: Add param
# valid_tag_names = ["Abfahrtsmonitor","Abfallbehälter-Abteil","Abfallbehälter-Abteil_fehlt","Abfallbehälter-Bahnhof","Abfallbehälter-Vorraum","Abfallbehälter_Graffiti","Abfalleimer-Bahnhof","Abfalleimer-Bahnhof_Graffiti","Abteil-Übergangstüre","Abteilwand_Graffiti","Analoge-Schliessfächer","Analoge-Schliessfächer_Verkratzt","Anderes-Piktogramm-Bahnhof","Ankunftsmonitor","Armlehne","Auto","Automatennummer","Bedien-und-Anzeigeelemente","Beleuchtung im Fst.","Beleuchtung-Bahnhof_AUS-defekt","Beleuchtung-Bahnhof_EIN","Beleuchtung-Wagen_AUS-defekt","Beleuchtung-Wagen_EIN","Betriebslagemonitor","Billettautomat-SBB","Billettautomat-SBB_Bildschirm-eingeschlagen","Billettautomat-SBB_Graffiti","Billettentwerter-SBB","Billettentwerter_Graffiti","Blaue-FIS-Tafel","Busnummer","CAB-Radio und Funkgerät","Dienstbeleuchtung-Schienenfz","Digitale-Schliessfächer","DMI_Bedien_Diagnosebildschirm","Einstiegsbereich-Wand_Graffiti","Einstiegstüre","Einstiegstüre_offen","ETCS","Fenster-innen-Wagen","Fenster-innen-Wagen_Graffiti","Feuerlöscher","FIS_Graffiti","Fluchthaube","Formulare","Fussboden-Bahnhof_beschädigt","Fussboden-Bahnhof_verschmutzt","Fussboden-Wagen_Verschmutzt","Fusspodest-Pedal","Füllstandsanz-Bioreaktor-Fäkalientank","Füllstandsanz-Wasser","Gebäudetüre","Gebäudetüre_Graffiti","Geldeinwurf","Generalanzeiger","Gleis-Piktogramm-Tafel","Handlauf-Bahnhof","Inventar Fzg.","Kamera-Wagen","KIS oder TIMS","KIS-Anzeige-aussen","KIS-Anzeige-innen","Klima und Heizung","Kupplung","Lautsprecher Fst.","Lautsprecher-Bahnhof","LEA-Halter und USB","Lift","Monitor-Wagen","Motorrad","Nothammer","Nothammer_fehlt","Pantograf","Parkautomat","Perronanzeiger","Perronsäule_Graffiti","Person","Pflanzenwuchs","Puffer-Stossvorrichtung","Quittierschalter","Re 460","Recyclingstation","Rolltreppe","Rollvorhang","Rückspiegel-und-Rücksehsysteme","Sander","SBB-Uhr","Scheibenwisch-und-Waschanlage","Schienenfz","Schienenfz_Graffiti","Schienenfz_Verschmutzung","Schlauch-Kupplung","Schnee","Seitenfenster Fst.","Sektor-Piktogramm-Tafel","Selecta-Kaffeeautomat","Selectaautomat","Selectaautomat_eingeschlagen","Sitz Lokführer","Sitz-Wagen","Sitz-Wagen_Kopfschutztuch-fehlt","Sitz-Wagen_Polster-aufgerissen","Sitz-Wagen_Polster-fehlt","Sitz-Wagen_Polster-verschmutzt","Sitzbank-Bahnhof","Sitzbank-Bahnhof_Graffiti","Sitzbank-Bahnhof_Verschmutzt","Sitzverstellung-Wagen","Sonnenschutz-Führerstand","Sonstige-Wagen-Graffitis","Sprechstelle-Zugpersonal","Steckdose-Wagen","Sticker","Text-auf-FIS-Tafel","Tisch-Wagen","Tisch-Wagen_Graffiti","Tisch-Wagen_Verschmutzt","Tor","Treppe-Bahnhof","Treppe-Bahnhof_Verschmutzt","Türe-Führerstand","Türgriff_Einstieg","Türknopf_Einstieg_schliessen","Türknopf_Einstieg_öffnen","UIC-Nummer","Umgebung_Graffiti","Unter-Überführung-Rampe_Graffiti","Velo","Wartesaal-Türe","WC-Bahnhof_Graffiti","WC-Kabine-Abfallbehälter","WC-Kabine-Defekt-Kleber","WC-Kabine-Oberfläche-Waschbereich-verkratzt","WC-Kabine-Seifenspender","WC-Kabine-Spiegel_Graffiti","WC-Kabine-Toilette","WC-Kabine-Türe","WC-Kabine-Türklinke-Griff","WC-Kabine-Türknopf","WC-Kabine-Wand_Graffiti","WC-Kabine-Waschbecken","WC-Kabine-Wasserhahn","WC-Türe-Bahnhof","ZUB"]

class SBB2Yolo:
    def __init__(self, coco_input_folders, yolo_output_folder, output_yolo_config_file,
                 dataset_split_pivot, overwrite=False):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.coco_input_folders = coco_input_folders
        self.yolo_output_folder = yolo_output_folder.rstrip('/')
        self.output_yolo_config_file = output_yolo_config_file
        self.dataset_split_pivot = dataset_split_pivot
        self.overwrite = overwrite

    def convert(self):
        """ Convert Coco Annotations from JSON file into YoloV5 Structure """

        output_path = Path(self.yolo_output_folder)

        if output_path.exists():
            if self.overwrite:
                shutil.rmtree(output_path)
            else:
                logging.info(f"\033[36mThe output dataset already exists ({self.yolo_output_folder}): not doing it again.\033[0m")
                return

        logging.info("\033[36mStarting conversion from Coco to Yolo format...\033[0m")
        # Create output folder structure
        self.images_train_out_folder, self.images_val_out_folder, self.labels_train_out_folder, self.labels_val_out_folder = FileSystemUtil.create_yolo_folder_structure(
            output_path)

        # Get all the labels
        annotation_names_all = []
        data = []
        for i, coco_input_folder in enumerate(self.coco_input_folders):
            logging.info(f"Reading from dataset: {coco_input_folder.rstrip('/')} ({i+1}/{len(self.coco_input_folders)})")
            coco, images_basedir, annotation_names, annotations_file = self.__init_coco(coco_input_folder)
            annotation_names_all.extend(annotation_names)
            data.append((coco, images_basedir, annotations_file))

        annotation_names_all = list(dict.fromkeys(annotation_names_all))  # Remove duplicate if any

        # TODO: Add param
        # annotation_names_all = [a for a in annotation_names_all if a in valid_tag_names]

        # Consider all coco input
        for i, (coco, images_basedir, annotations_file) in enumerate(data):
            logging.info(f"Converting dataset ({i+1}/{len(self.coco_input_folders)})")
            self.__convert_coco(coco, images_basedir, annotations_file, annotation_names_all)

        # Write yaml config file
        YamlUtil.write_yolo_config_file(
            str(self.images_train_out_folder), str(self.images_val_out_folder), annotation_names_all,
            output_path / self.output_yolo_config_file)
        logging.info(f"\033[32mConversion success\033[0m: saved in {self.yolo_output_folder}")
        logging.info(f"- The dataset contains {len(list(Path(self.images_train_out_folder).glob('**/*')))} images in train and {len(list(Path(self.images_val_out_folder).glob('**/*')))} images in val")
        logging.info(f"- There are {len(annotation_names_all)} labels")

    def __init_coco(self, coco_input_folder):
        input_path = Path(coco_input_folder)

        if not input_path.exists():
            logging.info(f"\033[31mConversion error:\033[0m the input path does not exist ({input_path})")
            exit(1)
        annotations_files = list(input_path.glob('**/*.json'))
        if not len(annotations_files) == 1:
            logging.info(
                f"\033[31mConversion error:\033[0m annotation files are not unique or are empty: {[str(a) for a in annotations_files]}")
            exit(1)
        annotations_file = annotations_files[0]
        coco = kwcoco.CocoDataset(str(annotations_file))
        images_basedir = Path(annotations_file).parents[0]
        annotation_names = [coco.cats[a]["name"] for a in coco.cats]
        logging.info(f"- There are {len(coco.imgs)} images")
        logging.info(f"- There are {len(annotation_names)} labels")
        return coco, images_basedir, annotation_names, annotations_file

    def __convert_coco(self, coco, images_basedir, annotations_file, annotation_names):
        nb_train = 0
        nb_val = 0
        # Consider all images
        for image_id in coco.imgs:
            file_name = coco.imgs[image_id]["file_name"]
            img_width = coco.imgs[image_id].get("width", 0)  # optional
            img_height = coco.imgs[image_id].get("height", 0)  # optional
            aids = coco.index.gid_to_aids[image_id]
            image_path = images_basedir / file_name

            # Determine whether to add it to train or val
            if random() < self.dataset_split_pivot:
                images_out_folder = self.images_train_out_folder
                labels_out_folder = self.labels_train_out_folder
                nb_train += 1
            else:
                images_out_folder = self.images_val_out_folder
                labels_out_folder = self.labels_val_out_folder
                nb_val += 1

            # TODO: use symlink instead?
            # Copy all images to target folder
            try:
                shutil.copyfile(image_path, images_out_folder / file_name)
            except FileNotFoundError:
                logging.info(f"\033[31mConversion error:\033[0m the image ({image_path}) does not exist.")
                exit(1)

            # Get the Yolo boxes
            yolo_boxes = []
            for aid in aids:
                coco_category_id = coco.anns[aid]["category_id"]
                annotation_name = coco.cats[coco_category_id]["name"]

                # TODO: Add param
                # if annotation_name not in valid_tag_names:
                #    print(f"skip {annotation_name}")
                #    continue

                x, y, w, h = coco.anns[aid]["bbox"]

                if img_height == 0.0 and img_width == 0.0:
                    (x_yolo, y_yolo, w_yolo, h_yolo) = convertRelCoco2Yolo(x, y, w, h)
                else:
                    (x_yolo, y_yolo, w_yolo, h_yolo) = convertAbsCoco2Yolo(
                        x, y, w, h, img_width, img_height)

                annotation_index = annotation_names.index(annotation_name)
                yolo_boxes.append(
                    [annotation_index, x_yolo, y_yolo, w_yolo, h_yolo])

            if len(yolo_boxes) == 0:
                print(
                    f"{file_name} has no annotations in {annotations_file}")
            else:
                txt_file_name = Path(file_name).with_suffix(".txt")
                TxtUtil.write_labels_txt_file(
                    yolo_boxes, labels_out_folder / txt_file_name)
        logging.info(f"- {nb_train} images are copied into train")
        logging.info(f"- {nb_val} images are copied into val")
        return annotation_names


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
    parser.add_argument('--overwrite', action='store_true', help="If set, will overwrite output dataset if it already exist.")
    # TODO 7: add argument include labels
    # TODO 8: add argument exclude labels
    # TODO 9: add argument for move and copy
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    SBB2Yolo(args.coco_input_folders, args.yolo_output_folder,
             args.yolo_config_file, args.dataset_split_pivot, args.overwrite).convert()
