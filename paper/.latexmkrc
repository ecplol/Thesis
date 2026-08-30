# Pin biber's PAR self-extraction cache to a stable location.
#
# biber is a PAR-packed Perl binary that unpacks its Perl library into a cache
# dir on first run. By default that lands in /var/folders/.../T/par-*, which
# macOS's tmp reaper prunes by last-access time (~3 days). Rarely-touched files
# such as the Unicode tables get deleted while the rest of the cache survives,
# leaving a half-populated cache. biber then dies mid-parse with
#
#   Unicode::UCD: failed to find unicore/version ... at Biber/Input/file/bibtex.pm
#
# and exit code 25 -- with no message in main.blg. latexmk reacts by deleting
# main.bbl and renaming main.bcf to main.bcf-SAVE-ERROR, so every later run
# fails with the misleading "main.bcf is malformed" until those are cleaned out.
# Net effect: the whole bibliography silently disappears every few days.
#
# Keeping the cache under $HOME puts it out of the reaper's reach.
$ENV{'PAR_GLOBAL_TEMP'} = "$ENV{HOME}/.cache/par-biber";
$ENV{'PAR_TEMP'}        = "$ENV{HOME}/.cache/par-biber";

# Break the self-perpetuating "main.bcf is malformed" loop.
#
# When biber fails for any reason -- including a transient read of a .bcf that
# another process (iCloud sync) is touching -- latexmk deletes main.bbl and
# quarantines main.bcf as main.bcf-SAVE-ERROR. Every LATER run then aborts on
# the leftover quarantine before it can regenerate anything, so main.pdf stays
# frozen at its last good build while you keep editing. Clearing the quarantine
# at startup means each run begins from a recoverable state.
unlink glob("*.bcf-SAVE-ERROR");
unlink glob("*.bbl-SAVE-ERROR");

# Always finish and emit a PDF, even if biber errors. A bibliography problem
# then costs you the citations for that run (they render as bold ?? marks)
# instead of silently blocking every text edit from reaching main.pdf.
$force_mode = 1;
