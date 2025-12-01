rsync -ruP \
  --include='*__complete*/' \
  --exclude='*' \
  myriad:/nfs/ghome/live/jwornbard/hudson/thinned_mfld/results/ \
  /home/zongchen/thinned_mfld/results_server/
