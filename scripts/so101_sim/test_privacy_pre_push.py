"""Regression tests for the pre-push privacy scanner. No real push or network calls."""
import importlib.util
from pathlib import Path
import unittest

script = Path(__file__).resolve().parents[1] / "privacy" / "check_pre_push.py"
spec = importlib.util.spec_from_file_location("privacy_pre_push", script)
privacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy)


def png_with_chunk(typ, payload=b""):
    import zlib
    def chunk(kind, dat):
        return len(dat).to_bytes(4, "big") + kind + dat + zlib.crc32(kind+dat).to_bytes(4,"big")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", b"1234567890123") + chunk(typ,payload) + chunk(b"IEND",b"")


class PrivacyPushTests(unittest.TestCase):
    def test_unix_path_and_username(self):
        issues=privacy.find_issues("reports/model.json",
            (b'{"source": "/' + b'Users/exampleperson/Documents/project", "token":"ok"}'),
            names={"exampleperson"})
        self.assertTrue(any("home path" in s for s in issues))
        self.assertTrue(any("username" in s for s in issues))

    def test_windows_path(self):
        issues=privacy.find_issues("report.txt", (b'C:\\' + b'Users\\myaccount\\Documents\\file'),
                                   names={"myaccount"})
        self.assertTrue(any("Windows" in s for s in issues))

    def test_clean_json(self):
        issues=privacy.find_issues("report.json", b'{"source":"assets/calibration/images"}',
                                   names={"exampleperson"})
        self.assertEqual(issues,[])

    def test_png_text_and_exif_blocked(self):
        for typ in (b"tEXt", b"zTXt", b"iTXt", b"eXIf"):
            issues=privacy.find_issues("calibration.png",png_with_chunk(typ,b"GPS data"),
                                       names=set())
            self.assertTrue(any("PNG metadata" in s for s in issues),typ)

    def test_png_no_metadata(self):
        issues=privacy.find_issues("calibration.png",png_with_chunk(b"IDAT",b"x"),
                                   names={"exampleperson"})
        self.assertEqual(issues,[])

    def test_jpeg_exif_blocked(self):
        issues=privacy.find_issues("photo.jpg",
                                   b"\xff\xd8\xff\xe1\x00\x12Exif\x00\x00GPS data",
                                   names=set())
        self.assertTrue(any("JPEG EXIF" in s for s in issues))

    def test_non_png_image_review(self):
        self.assertTrue(privacy.find_issues("photo.heic",b"whatever",names=set()))

    def test_parse_input(self):
        a="a"*40;b="b"*40
        self.assertEqual(privacy.parse_push_input(f"refs/heads/main {a} refs/heads/main {b}\n"),
                         [(a,b)])
        self.assertEqual(privacy.parse_push_input(f"refs/heads/main {a} refs/heads/new {privacy.ZERO}\n"),
                         [(a,None)])
        self.assertEqual(privacy.parse_push_input(f"refs/heads/main {privacy.ZERO} refs/heads/main {b}\n"),
                         [])


if __name__=="__main__":
    unittest.main()
