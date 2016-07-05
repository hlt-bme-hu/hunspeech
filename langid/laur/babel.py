import bob.db.verification.filelist
import bob.bio.base

custom_wav_directory = "[CUSTOM_WAV_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.verification.filelist.Database(
        'bob/bio/spear/config/database/custom/',
        original_directory = custom_wav_directory,
        original_extension = ".wav",),
    name = "custom",
    protocol = 'A',
)
