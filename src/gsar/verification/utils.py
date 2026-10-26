import cv2
import io
import base64
import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Optional, Union, Tuple, List, Dict
import dataclasses


@dataclasses.dataclass
class BoundingBox:
    """Class for representing a bounding box."""

    x_min: float | int
    x_max: float | int
    y_min: float | int
    y_max: float | int

    @property
    def center(self) -> tuple[float, float]:
        """Gets center of bounding box."""
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float | int:
        """Gets width of bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float | int:
        """Gets height of bounding box."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float | int:
        return self.width * self.height


@dataclasses.dataclass
class UIElement:
    """Represents a UI element."""

    text: Optional[str] = None  # 这个就是文本框里面的内容
    content_description: Optional[str] = None
    class_name: Optional[str] = None
    bbox: Optional[BoundingBox] = None  # 根据屏幕尺寸进行归一化的bbox
    bbox_pixels: Optional[BoundingBox] = None
    hint_text: Optional[str] = None
    is_checked: Optional[bool] = None
    is_checkable: Optional[bool] = None
    is_clickable: Optional[bool] = None
    is_editable: Optional[bool] = None
    is_enabled: Optional[bool] = None
    is_focused: Optional[bool] = None
    is_focusable: Optional[bool] = None
    is_long_clickable: Optional[bool] = None
    is_scrollable: Optional[bool] = None
    is_selected: Optional[bool] = None
    is_visible: Optional[bool] = None
    package_name: Optional[str] = None
    resource_name: Optional[str] = None
    tooltip: Optional[str] = None
    resource_id: Optional[str] = None

    element_id: Optional[str] = None
    # Make the actual storage field private and part of the dataclass init
    _image_hash: Optional[str] = dataclasses.field(
        default=None, repr=False
    )  # Use field to control repr if needed
    uid: Optional[str] = None

    def _get_element_id(self) -> str:
        """Generates a unique ID for the UI element based on its properties."""
        if self.element_id is None and self.bbox_pixels:
            elem_w, elem_h = self.bbox_pixels.width, self.bbox_pixels.height
            if self.resource_id:
                self.element_id = self.resource_id.replace(":", ".").replace("/", "_")
            else:
                self.element_id = f"{self.class_name}_{elem_w}_{elem_h}"
            if self.content_description and len(self.content_description) < 20:
                content_desc = (
                    self.content_description.replace("/", "_")
                    .replace(" ", "")
                    .replace(":", "_")
                )
                self.element_id += f"_{content_desc}"
        return self.element_id

    def _maybe_is_editable(self) -> bool:
        """Checks if the UI element is editable based on its properties."""
        if self.is_editable:
            return True
        if self.is_clickable and self.class_name:
            for s in [
                "EditText",
                "TextView",
                "AutoCompleteTextView",
                "MultiAutoCompleteTextView",
            ]:
                if s in self.class_name:
                    return True
        return False

    def _update_uid(self) -> None:  # WARNING:在更新image_hash之后记得调用这个函数
        if self.element_id is not None and self._image_hash is not None:
            self.uid = f"{self.element_id}_{self._image_hash }"
        elif self._image_hash is not None:
            self.uid = f"{self._image_hash }"
        elif self.element_id is not None:
            self.uid = f"{self.element_id}"

    def set_image_hash(self, image: Image.Image) -> None:
        """Set the image hash for the UI element."""
        self.image_hash = str(
            imagehash.phash(image, hash_size=16, highfreq_factor=8)
        ).upper()

    # Define the property getter and setter
    @property
    def image_hash(self) -> Optional[str]:
        """Getter for the image hash."""
        return self._image_hash

    # 用setter实现更新image_hash之后自动更新uid
    @image_hash.setter
    def image_hash(self, value: Optional[str]) -> None:
        """Setter for the image hash that also updates the UID."""
        if self._image_hash != value:
            self._image_hash = value
            self._update_uid()

    def __post_init__(self):
        """Post-initialization to ensure element_id is set."""
        if self.bbox_pixels is not None and isinstance(self.bbox_pixels, dict):
            self.bbox_pixels = BoundingBox(**self.bbox_pixels)
        if self.bbox is not None and isinstance(self.bbox, dict):
            self.bbox = BoundingBox(**self.bbox)
        self.element_id = self._get_element_id()
        self.is_editable = self._maybe_is_editable()
        self._update_uid()



def dict_to_ui_element(el_dict: Dict) -> UIElement:
    """
    Convert a dictionary representing a UI element into a UIElement instance.

    Args:
        el_dict (dict): Dictionary with keys matching UIElement fields

    Returns:
        UIElement: Instance of the UIElement class
    """
    # 处理 bbox_pixels
    bbox_pixels = None
    if el_dict.get("bbox_pixels"):
        bbox_pixels = BoundingBox(**el_dict["bbox_pixels"])

    # 处理 bbox（归一化 bbox，如果有）
    bbox = None
    if el_dict.get("bbox"):
        bbox = BoundingBox(**el_dict["bbox"])

    # 创建 UIElement
    ui_element = UIElement(
        text=el_dict.get("text"),
        content_description=el_dict.get("content_description"),
        class_name=el_dict.get("class_name"),
        bbox=bbox,
        bbox_pixels=bbox_pixels,
        hint_text=el_dict.get("hint_text"),
        is_checked=el_dict.get("is_checked"),
        is_checkable=el_dict.get("is_checkable"),
        is_clickable=el_dict.get("is_clickable"),
        is_editable=el_dict.get("is_editable"),
        is_enabled=el_dict.get("is_enabled"),
        is_focused=el_dict.get("is_focused"),
        is_focusable=el_dict.get("is_focusable"),
        is_long_clickable=el_dict.get("is_long_clickable"),
        is_scrollable=el_dict.get("is_scrollable"),
        is_selected=el_dict.get("is_selected"),
        is_visible=el_dict.get("is_visible"),
        package_name=el_dict.get("package_name"),
        resource_name=el_dict.get("resource_name"),
        tooltip=el_dict.get("tooltip"),
        resource_id=el_dict.get("resource_id"),
    )
    return ui_element


def _validate_ui_element(
    ui_element: UIElement,
    screen_width_height_px: tuple[int, int],
) -> bool:
    screen_width, screen_height = screen_width_height_px

    # Filters out invisible element.
    if not ui_element.is_visible:
        return False

    # Filters out element with invalid bounding box.
    if ui_element.bbox_pixels:
        x_min = ui_element.bbox_pixels.x_min
        x_max = ui_element.bbox_pixels.x_max
        y_min = ui_element.bbox_pixels.y_min
        y_max = ui_element.bbox_pixels.y_max

        if (
            x_min >= x_max
            or x_min >= screen_width
            or x_max <= 0
            or y_min >= y_max
            or y_min >= screen_height
            or y_max <= 0
        ):
            return False

    return True


def _format_element(ui_element: UIElement, index):
    if ui_element.class_name:
        _class = f" class=\"{ui_element.class_name.split('.')[-1]}\""
    else:
        _class = ''
    
    text = ui_element.text
    if text is None:
        text = ui_element.content_description
    if text is None:
        text = ''
    
    if ui_element.tooltip:
        tooltip = f' tooltip="{ui_element.tooltip}"'
    else:
        tooltip = ''
    
    if ui_element.resource_name:
        if 'inputmethod' in ui_element.resource_name:
            return None
        resource_name = f' resource-name="{ui_element.resource_name}"'
    else:
        resource_name = ''
    
    attribute = ''
    
    if ui_element.is_clickable:
        attribute += ' clickable'
        
    if ui_element.is_checkable:
        attribute += ' checkable'
        
        if ui_element.is_checked:
            attribute += ' status="on"'
        else:
            attribute += ' status="off"'
        
    if ui_element.is_editable:
        attribute += ' editable'
    
    return f'<element id="{index}"{_class}{tooltip}{resource_name}{attribute}> {text} </element>'


def _generate_ui_elements_description_list_full(
    ui_elements: List[UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
    """Generate description for a list of UIElement using full information.

    Args:
        ui_elements: UI elements for the current screen.
        screen_width_height_px: Logical screen size.

    Returns:
        Information for each UIElement.
    """
    tree_info = ''
    for index, ui_element in enumerate(ui_elements):
        if _validate_ui_element(ui_element, screen_width_height_px):
            formated_element = _format_element(ui_element, index)
            if formated_element:
                tree_info += formated_element + '\n'
    
    return tree_info.strip()


def _logical_to_physical(
    logical_coordinates: tuple[int, int],
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
) -> tuple[int, int]:
    """Convert logical coordinates to physical coordinates.

    Args:
      logical_coordinates: The logical coordinates for the point.
      logical_screen_size: The logical screen size.
      physical_frame_boundary: The physical coordinates in portrait orientation
        for the upper left and lower right corner for the frame.
      orientation: The current screen orientation.

    Returns:
      The physical coordinate for the point in portrait orientation.

    Raises:
      ValueError: If the orientation is not valid.
    """
    x, y = logical_coordinates
    px0, py0, px1, py1 = physical_frame_boundary
    px, py = px1 - px0, py1 - py0
    lx, ly = logical_screen_size
    if orientation == 0:
        return (int(x * px / lx) + px0, int(y * py / ly) + py0)
    if orientation == 1:
        return (px - int(y * px / ly) + px0, int(x * py / lx) + py0)
    if orientation == 2:
        return (px - int(x * px / lx) + px0, py - int(y * py / ly) + py0)
    if orientation == 3:
        return (int(y * px / ly) + px0, py - int(x * py / lx) + py0)
    print("Invalid orientation.")
    raise ValueError("Unsupported orientation.")


def _ui_element_logical_corner(
    ui_element: UIElement, orientation: int
) -> list[tuple[int, int]]:
    """Get logical coordinates for corners of a given UI element.

    Args:
      ui_element: The corresponding UI element.
      orientation: The current orientation.

    Returns:
      Logical coordinates for upper left and lower right corner for the UI
      element.

    Raises:
      ValueError: If bounding box is missing.
      ValueError: If orientation is not valid.
    """
    if ui_element.bbox_pixels is None:
        raise ValueError("UI element does not have bounding box.")
    if orientation == 0:
        return [
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
        ]
    if orientation == 1:
        return [
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
        ]
    if orientation == 2:
        return [
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
        ]
    if orientation == 3:
        return [
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
        ]
    raise ValueError("Unsupported orientation.")


def add_ui_element_mark(
    screenshot: np.ndarray,
    ui_element: UIElement,
    index: int | str,
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
):
    """Add mark (a bounding box plus index) for a UI element in the screenshot.

    Args:
      screenshot: The screenshot as a numpy ndarray.
      ui_element: The UI element to be marked.
      index: The index for the UI element.
      logical_screen_size: The logical screen size.
      physical_frame_boundary: The physical coordinates in portrait orientation
        for the upper left and lower right corner for the frame.
      orientation: The current screen orientation.
    """
    if ui_element.bbox_pixels:
        upper_left_logical, lower_right_logical = _ui_element_logical_corner(
            ui_element, orientation
        )
        upper_left_physical = _logical_to_physical(
            upper_left_logical,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
        lower_right_physical = _logical_to_physical(
            lower_right_logical,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

        cv2.rectangle(
            screenshot,
            upper_left_physical,
            lower_right_physical,
            color=(0, 255, 0),
            # thickness=2,
            thickness=3,
        )
        screenshot[
            upper_left_physical[1] + 1 : upper_left_physical[1] + 25,
            upper_left_physical[0] + 1 : upper_left_physical[0] + 35,
            :,
        ] = (255, 255, 255)
        cv2.putText(
            screenshot,
            str(index),
            (
                upper_left_physical[0] + 1,
                upper_left_physical[1] + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            thickness=2,
            # thickness=3,#这个会导致标签的数字变成一团
        )


def draw_red_border(image, coordinates_list, border_width=5):
    """
    Draw red borders around specified rectangles on an image.

    :param image: PIL.Image.Image - The input image.
    :param coordinates_list: list - A list of tuples, each containing four integers (x1, y1, x2, y2) representing the coordinates of the rectangles.
    :param border_width: int - The width of the red border (default is 5 pixels).
    """
    draw = ImageDraw.Draw(image)
    
    # Draw red borders for each set of coordinates
    for coordinates in coordinates_list:
        x1, y1, x2, y2 = coordinates
        
        # Adjust the coordinates to draw the border outside the crop area
        x1 -= border_width // 2
        y1 -= border_width // 2
        x2 += border_width // 2
        y2 += border_width // 2
        
        # Draw the red border
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=border_width)
    
    # # Save the image
    # image.save(save_path)
    return image


def draw_img_name(image, text, font_size=60):
    draw = ImageDraw.Draw(image)
    W, H = image.size

    bar_width_ratio = 2 / 5

    # ---------- 字体 ----------
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # ---------- 文本尺寸 ----------
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_y = int(font_size * 0.4)
    bar_h = text_h + padding_y * 2

    # ---------- 背景条（1/3 宽） ----------
    bar_w = int(W * bar_width_ratio)
    bar_x1 = (W - bar_w) // 2
    bar_x2 = bar_x1 + bar_w

    draw.rectangle(
        [(bar_x1, 0), (bar_x2, bar_h)],
        # fill=(255, 255, 255, 255)  # white
        fill = (0, 0, 0, 255)   # black
    )

    # ---------- 文本居中 ----------
    text_x = (W - text_w) // 2
    text_y = padding_y

    draw.text(
        (text_x, text_y),
        text,
        fill=(255, 255, 255, 255),  # white
        font=font
    )

    return image


def encode_image_to_data_url(image: Union[str, Image.Image], format="PNG"):
    if isinstance(image, str):
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
    else:
        raise TypeError("image must be a file path or PIL.Image.Image")
    
    return f"data:image/{format.lower()};base64,{b64}"

