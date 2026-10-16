"""CVAT for images 1.1 XML dataset import, not a REST annotation payload."""
import math
import xml.etree.ElementTree as ET


def export_cvat_xml(records, conf_thresh=0.0):
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'
    labels = sorted({s['label'] for r in records for s in r['shapes']
                     if s.get('confidence',1.) >= conf_thresh})
    task = ET.SubElement(ET.SubElement(root, 'meta'), 'task')
    ET.SubElement(task, 'name').text = 'M38 reviewed prelabels'
    ET.SubElement(task, 'size').text = str(len(records))
    ET.SubElement(task, 'mode').text = 'annotation'
    label_node = ET.SubElement(task, 'labels')
    for label in labels:
        node = ET.SubElement(label_node, 'label')
        ET.SubElement(node, 'name').text = label
        ET.SubElement(node, 'type').text = 'rectangle'
        ET.SubElement(node, 'attributes')
    for index, record in enumerate(records):
        width, height = int(record['width']), int(record['height'])
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions must be positive')
        image = ET.SubElement(root,'image',id=str(index),name=record['image'],width=str(width),height=str(height))
        for shape in record['shapes']:
            if shape.get('confidence',1.) < conf_thresh:
                continue
            x1,y1,x2,y2 = map(float, shape['points'])
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('Invalid or out-of-image rectangle')
            ET.SubElement(image,'box',label=shape['label'],occluded='0',
                source='manual' if shape.get('edited') else 'auto',
                xtl=str(x1),ytl=str(y1),xbr=str(x2),ybr=str(y2),z_order='0')
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root,encoding='unicode')
