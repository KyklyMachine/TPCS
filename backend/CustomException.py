import json
import warnings


class CustomException:
    _msg_dict = {}

    def __init__(self, msg_path: str):
        try:
            with open(msg_path) as json_file:
                self._msg_dict = json.load(json_file)
        except FileNotFoundError as e:
            print(e, "MSG_CONSTANT.json NOT FOUND!")

    def update_exception(self, exception: Exception, msg_name: str) -> Exception:
        add_msg = ""
        try:
            add_msg = self._msg_dict[msg_name]
        except KeyError:
            warnings.warn(f"Key '{msg_name}' not found")
        return type(exception)(str(add_msg) + str(exception))
