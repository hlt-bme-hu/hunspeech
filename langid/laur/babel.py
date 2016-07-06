import bob.db.verification.filelist
import bob.bio.base

# babel_wav_directory = "[YOUR_HUNSPEECH_WAV_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.verification.filelist.Database(
        # TODO Later replace this first argument by
        # pkg_resources.resource_filename(
        #     'bob.db.hunspeech', 'config/database/hunspeech')
        '/mnt/store/hlt/Speech/LangPack/bob_list/', # Contains the file lists
        original_directory = '/mnt/store/hlt/Speech/LangPack',
        original_extension = ".wav",),
    name = "babel",
    protocol = '', # There is only a single protocol
)
