
import re
import copy
import time
import logging
from typing import Any, Optional

from ...common.device import Device
from . import json_action
logger = logging.getLogger(__name__)

"""
This file contains utility functions for interacting with Android devices using ADB.
"""

def send_android_intent(
    command: str,
    action: str,
    device_controller: Device,
    data_uri: str | None = None,
    mime_type: str | None = None,
    extras: dict[str, Any] | None = None,
    timeout_sec: int = 10,
):
    """Sends an intent to Android device using adb.

    This is a low-level command for sending an intent with additional parameters.
    When these additional parameters are not necessary, consider instead using
    `adb_utils.start_activity()` or `env.execute_adb_call()` with
    `AdbRequest.StartActivity` or `AdbRequest.SendBroadcast`.

    Args:
      command: Either "start" for start activity intents or "broadcast" for
        broadcast intents.
      action: The broadcast action (e.g. "android.intent.action.VIEW").
      device_controller: The environment to which the broadcast is sent.
      data_uri: Optional intent data URI (e.g. "content://contacts/people/1").
      mime_type: Optional mime type (e.g. "image/png").
      extras: Dictionary containing keys and values to be sent as extras.
      timeout_sec: The maximum time in seconds to wait for the broadcast to
        complete.

    Returns:
      AdbResponse object.
    """
    if command not in ["start", "broadcast"]:
        raise ValueError('Intent command must be either "start" or "broadcast"')

    # adb_command = ["shell", "am", command, "-a", action]
    adb_command = ["am", command, "-a", action]

    if data_uri:
        adb_command.extend(["-d", f'"{data_uri}"'])

    if mime_type:
        adb_command.extend(["-t", f'"{mime_type}"'])

    if extras:
        for key, value in extras.items():
            if isinstance(value, tuple):
                type_override, value = value
                if type_override == "str":
                    adb_command.extend(["--es", key, f'"{value}"'])
                elif type_override == "bool":
                    adb_command.extend(["--ez", key, f'"{value}"'])
                elif type_override == "int":
                    adb_command.extend(["--ei", key, f'"{value}"'])
                elif type_override == "long":  # long type only available via override.
                    adb_command.extend(["--el", key, f'"{value}"'])
                elif type_override == "float":
                    adb_command.extend(["--ef", key, f'"{value}"'])
                elif type_override == "string array":
                    array_str = ",".join(value)
                    adb_command.extend(["--esa", key, f'"{array_str}"'])
            elif isinstance(value, str):
                adb_command.extend(["--es", key, f'"{value}"'])
            elif isinstance(value, bool):
                adb_command.extend(["--ez", key, f'"{value}"'])
            elif isinstance(value, int):
                adb_command.extend(["--ei", key, f'"{value}"'])
            # long type only available via override above.
            elif isinstance(value, float):
                adb_command.extend(["--ef", key, f'"{value}"'])
            elif isinstance(value, list):
                array_str = ",".join(value)
                adb_command.extend(["--esa", key, f'"{array_str}"'])
            else:
                raise ValueError(f"Unrecognized extra type for {key}")

    return device_controller.run_shell_command(adb_command, timeout=timeout_sec)


def display_message(message: str, device_controller: Device, header: str = "") -> None:
    send_android_intent(
        command="broadcast",
        action="com.example.ACTION_UPDATE_OVERLAY",
        device_controller=device_controller,
        extras={"task_type_string": header, "goal_string": message},
    )


def execute_adb_action(
    action: json_action.JSONAction,
    device_controller: Device,
    screen_elements: list[Any] = None,  # list[UIElement]
    screen_size: tuple[int, int] = None,  # (width, height)
) -> tuple[int, ...] | None:
    """Execute an action based on a JSONAction object.

    Args:
        action: JSONAction object containing the action to be executed.
        screen_elements: List of UI elements on the screen.
        screen_size: The (width, height) of the screen.
        env: The environment to execute the action in.
    """
    if action.action_type == json_action.ANSWER:
        if action.text:
            display_message(
                action.text,
                header="Agent answered:",
                device_controller=device_controller,
            )
        return
    if action.action_type in ["click", "double_tap", "long_press"]:
        idx = action.index
        x = action.x
        y = action.y
        if idx is not None and screen_elements is not None:
            if idx < 0 or idx >= len(screen_elements):
                raise ValueError(
                    f"Invalid element index: {idx}, must be between 0 and"
                    f" {len(screen_elements)-1}."
                )
            element = screen_elements[idx]
            if element.bbox_pixels is None:
                raise ValueError("Bbox is not present on element.")
            x, y = element.bbox_pixels.center
            x, y = int(x), int(y)
            if action.action_type == "click":
                device_controller.click(x, y)
            elif action.action_type == "double_tap":
                device_controller.double_click(x, y)
            else:
                device_controller.long_press(x, y)
        elif x is not None and y is not None:
            if action.action_type == "click":
                device_controller.click(x, y)
            elif action.action_type == "double_tap":
                device_controller.double_click(x, y)
            else:
                device_controller.long_press(x, y)
        else:
            raise ValueError(f"Invalid click action: {action}")
        return x, y

    elif action.action_type == "input_text":
        text = action.text
        if text:
            # First focus on enter text UI element.
            click_action = copy.deepcopy(action)
            click_action.action_type = "click"
            coordinates = execute_adb_action(
                action=click_action,
                device_controller=device_controller,
                screen_elements=screen_elements,
                screen_size=screen_size,
            )
            time.sleep(1.0)
            device_controller.input_text(text, smart_enter=True, clear_first=True)
            return coordinates
        else:
            logging.warning(
                "Input_text action indicated, but no text provided. No "
                "action will be executed."
            )

    elif action.action_type == "keyboard_enter":
        device_controller.enter()

    elif action.action_type == "navigate_home":
        device_controller.home()

    elif action.action_type == "navigate_back":
        device_controller.back()

    elif action.action_type == "scroll":
        if action.start_position is not None and action.end_position is not None:
            start_x, start_y = action.start_position
            end_x, end_y = action.end_position
        else:
            screen_width, screen_height = screen_size
            if action.index is not None:
                x_min, y_min, x_max, y_max = (
                    max(screen_elements[action.index].bbox_pixels.x_min, 0),
                    max(screen_elements[action.index].bbox_pixels.y_min, 0),
                    min(screen_elements[action.index].bbox_pixels.x_max, screen_width),
                    min(screen_elements[action.index].bbox_pixels.y_max, screen_height),
                )
            else:
                x_min, y_min, x_max, y_max = (0, 0, screen_width, screen_height)

            start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            direction = action.direction
            if direction == "down":
                end_x, end_y = (x_min + x_max) // 2, y_min
            elif direction == "up":
                end_x, end_y = (x_min + x_max) // 2, y_max
            elif direction == "right":
                end_x, end_y = x_min, (y_min + y_max) // 2
            elif direction == "left":
                end_x, end_y = x_max, (y_min + y_max) // 2
            else:
                print("Invalid direction")
                return
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
        return int(start_x), int(start_y), int(end_x), int(end_y)

    elif action.action_type == "swipe":  # Inverse of scroll.
        if action.start_position is not None and action.end_position is not None:
            start_x, start_y = action.start_position
            end_x, end_y = action.end_position
        else:
            screen_width, screen_height = screen_size
            mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
            direction = action.direction
            if direction == "down":
                start_x, start_y = mid_x, 0
                end_x, end_y = mid_x, screen_height
            elif direction == "up":
                start_x, start_y = mid_x, screen_height
                end_x, end_y = mid_x, 0
            elif direction == "left":
                start_x, start_y = 0, mid_y
                end_x, end_y = screen_width, mid_y
            elif direction == "right":
                start_x, start_y = screen_width, mid_y
                end_x, end_y = 0, mid_y
            else:
                print("Invalid direction")
                return
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
        return int(start_x), int(start_y), int(end_x), int(end_y)

    elif action.action_type == "open_app":
        app_name = action.app_name
        if app_name:
            launch_app(app_name, device_controller)
        else:
            raise ValueError("No app name provided")

    elif action.action_type == "wait":
        time.sleep(2.0)

    elif action.action_type == "launch_adb_activity":
        if action.activity_nickname == "app_drawer":
            device_controller.home()
            time.sleep(1.0)
            start_x, start_y = int(screen_size[0] / 2), int(screen_size[1] * 0.9)
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            device_controller.swipe(start_x, start_y, end_x, end_y)
        elif action.activity_nickname == "quick_settings":
            start_x, start_y = int(screen_size[0] / 2), 30
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            device_controller.swipe(start_x, start_y, end_x, end_y, duration=0.1)
    elif action.action_type == "change_orientation":
        change_orientation(action.orientation, device_controller)
    elif action.action_type == json_action.UNKNOWN:
        print("Unknown action type; no action will be executed. Try again...")
    else:
        print("Invalid action type")


def launch_app(app_name: str, device_controller: Device) -> Optional[str]:
    """Uses regex and ADB activity to try to launch an app.

    Args:
      app_name: The name of the app, as represented as a key in
        _PATTERN_TO_ACTIVITY.
      device_controller: The device controller to execute the command.

    Returns:
      The name of the app that is launched.
    """
    # Maps app names to the activity that should be launched to open the app.
    _PATTERN_TO_ACTIVITY = {
        "google chrome|chrome": (
            "com.android.chrome/com.google.android.apps.chrome.Main"
        ),
        "google chat": "com.google.android.apps.dynamite/com.google.android.apps.dynamite.startup.StartUpActivity",
        "settings|system settings": "com.android.settings/.Settings",
        "youtube|yt": "com.google.android.youtube/com.google.android.apps.youtube.app.WatchWhileActivity",
        "google play|play store|gps": (
            "com.android.vending/com.google.android.finsky.activities.MainActivity"
        ),
        "gmail|gemail|google mail|google email|google mail client": (
            "com.google.android.gm/.ConversationListActivityGmail"
        ),
        "google maps|gmaps|maps|google map": (
            "com.google.android.apps.maps/com.google.android.maps.MapsActivity"
        ),
        "google photos|gphotos|photos|google photo|google pics|google images": "com.google.android.apps.photos/com.google.android.apps.photos.home.HomeActivity",
        "google calendar|gcal": (
            "com.google.android.calendar/com.android.calendar.AllInOneActivity"
        ),
        "camera": "com.android.camera2/com.android.camera.CameraLauncher",
        "audio recorder": "com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.welcome.WelcomeActivity",
        "google drive|gdrive|drive": (
            "com.google.android.apps.docs/.drive.startup.StartupActivity"
        ),
        "google keep|gkeep|keep": (
            "com.google.android.keep/.activities.BrowseActivity"
        ),
        "grubhub": (
            "com.grubhub.android/com.grubhub.dinerapp.android.splash.SplashActivity"
        ),
        "tripadvisor": "com.tripadvisor.tripadvisor/com.tripadvisor.android.ui.launcher.LauncherActivity",
        "starbucks": "com.starbucks.mobilecard/.main.activity.LandingPageActivity",
        "google docs|gdocs|docs": "com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "google sheets|gsheets|sheets": "com.google.android.apps.docs.editors.sheets/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "google slides|gslides|slides": "com.google.android.apps.docs.editors.slides/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "clock": "com.google.android.deskclock/com.android.deskclock.DeskClock",
        "google search|google": "com.google.android.googlequicksearchbox/com.google.android.googlequicksearchbox.SearchActivity",
        "contacts": "com.google.android.contacts/com.android.contacts.activities.PeopleActivity",
        "facebook|fb": "com.facebook.katana/com.facebook.katana.LoginActivity",
        "whatsapp|wa": "com.whatsapp/com.whatsapp.Main",
        "instagram|ig": (
            "com.instagram.android/com.instagram.mainactivity.MainActivity"
        ),
        "twitter|tweet": "com.twitter.android/com.twitter.app.main.MainActivity",
        "snapchat|sc": "com.snapchat.android/com.snap.mushroom.MainActivity",
        "telegram|tg": "org.telegram.messenger/org.telegram.ui.LaunchActivity",
        "linkedin": (
            "com.linkedin.android/com.linkedin.android.authenticator.LaunchActivity"
        ),
        "spotify|spot": "com.spotify.music/com.spotify.music.MainActivity",
        "netflix": "com.netflix.mediaclient/com.netflix.mediaclient.ui.launch.UIWebViewActivity",
        "amazon shopping|amazon|amzn": (
            "com.amazon.mShop.android.shopping/com.amazon.mShop.home.HomeActivity"
        ),
        "tiktok|tt": "com.zhiliaoapp.musically/com.ss.android.ugc.aweme.splash.SplashActivity",
        "discord": "com.discord/com.discord.app.AppActivity$Main",
        "reddit": "com.reddit.frontpage/com.reddit.frontpage.MainActivity",
        "pinterest": "com.pinterest/com.pinterest.activity.PinterestActivity",
        "android world": "com.example.androidworld/.MainActivity",
        "files": "com.google.android.documentsui/com.android.documentsui.files.FilesActivity",
        "markor": "net.gsantner.markor/net.gsantner.markor.activity.MainActivity",
        "clipper": "ca.zgrs.clipper/ca.zgrs.clipper.Main",
        "messages": "com.google.android.apps.messaging/com.google.android.apps.messaging.ui.ConversationListActivity",
        "simple sms messenger|simple sms|sms messenger": "com.simplemobiletools.smsmessenger/com.simplemobiletools.smsmessenger.activities.MainActivity",
        "dialer|phone": "com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity",
        "simple calendar pro|simple calendar": "com.simplemobiletools.calendar.pro/com.simplemobiletools.calendar.pro.activities.MainActivity",
        "simple gallery pro|simple gallery": "com.simplemobiletools.gallery.pro/com.simplemobiletools.gallery.pro.activities.MainActivity",
        "miniwob": "com.google.androidenv.miniwob/com.google.androidenv.miniwob.app.MainActivity",
        "simple draw pro": "com.simplemobiletools.draw.pro/com.simplemobiletools.draw.pro.activities.MainActivity",
        "pro expense|pro expense app": (
            "com.arduia.expense/com.arduia.expense.ui.MainActivity"
        ),
        "broccoli|broccoli app|broccoli recipe app|recipe app": (
            "com.flauschcode.broccoli/com.flauschcode.broccoli.MainActivity"
        ),
        "caa|caa test|context aware access": "com.google.ccc.hosted.contextawareaccess.thirdpartyapp/.ChooserActivity",
        "osmand": "net.osmand/net.osmand.plus.activities.MapActivity",
        "tasks|tasks app|tasks.org:": (
            "org.tasks/com.todoroo.astrid.activity.MainActivity"
        ),
        "open tracks sports tracker|activity tracker|open tracks|opentracks": (
            "de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity"
        ),
        "joplin|joplin app": "net.cozic.joplin/.MainActivity",
        "vlc|vlc app|vlc player": "org.videolan.vlc/.gui.MainActivity",
        "retro music|retro|retro player": (
            "code.name.monkey.retromusic/.activities.MainActivity"
        ),
    }

    def get_adb_activity(app_name: str) -> Optional[str]:
        """Get a mapping of regex patterns to ADB activities top Android apps."""
        for pattern, activity in _PATTERN_TO_ACTIVITY.items():
            if re.match(pattern.lower(), app_name.lower()):
                return activity

    # Special app names that will trigger opening the default app.
    _DEFAULT_URIS: dict[str, str] = {
        "calendar": "content://com.android.calendar",
        "browser": "http://",
        "contacts": "content://contacts/people/",
        "email": "mailto:",
        "gallery": "content://media/external/images/media/",
    }
    if app_name in _DEFAULT_URIS:
        data_uri = _DEFAULT_URIS[app_name]
        adb_command = [
            #'shell',
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            data_uri,
        ]
        device_controller.run_shell_command(adb_command)
        time.sleep(1.0)  # Wait for the app to launch.
        return app_name

    activity = get_adb_activity(app_name)
    if activity is None:
        logging.error("Failed to launch app: %r", app_name)
        return None
    # start_activity(activity, extra_args=[], env=env, timeout_sec=5)
    args = [
        "am",
        "start",
        "-a",
        "android.intent.action.MAIN",
        "-c",
        "android.intent.category.LAUNCHER",
        "-n",
        activity,
    ]
    device_controller.run_shell_command(args)
    logging.info("Launched app: %r", app_name)
    time.sleep(1.0)  # Wait for the app to launch.
    return app_name


def change_orientation(orientation: str, device_controller: Device) -> None:
    """Changes the screen orientation.

    Args:
      orientation: str, The new orientation. Can be portrait, landscape,
        reverse_portrait, or reverse_landscape.
      device_controller: The device controller to execute the command.

    Raises:
      ValueError if invalid orientation is provided.
    """
    _ORIENTATIONS = {
        "portrait": "0",
        "landscape": "1",
        "portrait_reversed": "2",
        "landscape_reversed": "3",
    }
    if orientation not in _ORIENTATIONS:
        raise ValueError(
            f"Unknown orientation provided: {orientation} not in"
            f" {_ORIENTATIONS.keys()}"
        )
    command = [
        # "shell",
        "settings",
        "put",
        "system",
    ]
    # Turn off accelerometer.
    device_controller.run_shell_command(command + ["accelerometer_rotation", "0"])
    device_controller.run_shell_command(
        command + ["user_rotation", _ORIENTATIONS[orientation]]
    )
