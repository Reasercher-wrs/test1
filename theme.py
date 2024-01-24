from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable

class Style(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.neutral,
        secondary_hue: colors.Color | str = colors.neutral,
        neutral_hue: colors.Color | str = colors.neutral,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (fonts.GoogleFont("Sora")),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (fonts.GoogleFont("Sora")),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="#f8f9fa",  # Very light gray background color
            background_fill_primary_dark="#f8f9fa",  # Very light gray background color
            background_fill_secondary="#e9ecef",  # Light gray background color
            background_fill_secondary_dark="#e9ecef",  # Light gray background color
            block_background_fill="#f8f9fa",  # Very light gray background color
            block_background_fill_dark="#f8f9fa",  # Very light gray background color
            
            border_color_primary="#ced4da",  # Light border color
            border_color_primary_dark="#ced4da",  # Light border color
            
            link_text_color="#6c757d",  # Subdued gray link color
            link_text_color_dark="#6c757d",  # Subdued gray link color
            
            block_info_text_color="#212529",  # Dark text color
            block_info_text_color_dark="#212529",  # Dark text
            
            block_border_color="#ced4da",  # Light border color
            block_border_color_dark="#ced4da",  # Light border color
            block_shadow="*shadow_drop_lg",
            
            input_background_fill="#ffffff",  # Light background color
            input_background_fill_dark="#ffffff",  # Light background color
            input_border_color="#ced4da",  # Light border color
            input_border_color_dark="#ced4da",  # Light border color
            input_border_width="2px",
            
            block_label_background_fill="#f8f9fa",  # Very light gray background color
            block_label_background_fill_dark="#f8f9fa",  # Very light gray background color
            block_label_text_color="#212529",  # Dark text color
            block_label_text_color_dark="#212529",  # Dark text color
            
            button_primary_background_fill="#343a40",  # Dark gray background color
            button_primary_border_color_dark="#343a40",  # Dark gray border color
            button_primary_text_color="white",  # Light text color
            button_secondary_text_color_dark="black",  # Light text color
            button_shadow="*shadow_drop_lg",
            
            block_title_background_fill="#f8f9fa",  # Very light gray background color
            block_title_background_fill_dark="#f8f9fa",  # Very light gray background color
            block_title_radius="*radius_sm",
            block_title_text_color="#212529",  # Dark text color
            block_title_text_color_dark="#212529",  # Dark text color
            block_title_text_size="*text_lg",
            block_title_border_width="0px",  # Border width
            block_title_border_width_dark="0px",  # Border width
            block_title_border_color="#ced4da",  # Light border color
            block_title_border_color_dark="#ced4da",  # Light border color
            block_title_text_weight="600",
            
            body_background_fill="#f8f9fa",  # Very light gray background color
            body_background_fill_dark="#f8f9fa",  # Very light gray background color
            body_text_color="#212529",  # Dark text color
            body_text_color_dark="#212529",  # Dark text color
            body_text_color_subdued="#212529",  # Subdued gray text color
            body_text_color_subdued_dark="#212529",  # Subdued gray text color
            
            slider_color="#212529",  # Subdued gray slider color
            slider_color_dark="#212529",  # Subdued gray slider color

    
        )