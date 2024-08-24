def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None

@app.post("/api/upload-audio")
async def upload_audio(request):
    # Extract the file name and audio data from the request
    file_name = request.form.get("file_name")
    audio_data = request.form.get("audio_data")
    # print(file_name)
    # print(audio_data)

    if file_name and audio_data:
        # Append ".wav" to the file name
        file_name = file_name + ".wav"

        # Construct the file path
        file_path = os.path.join("voices", file_name)

        # Decode the base64 audio data
        audio_bytes = base64.b64decode(audio_data)

        # Save the audio bytes as a WAV file
        with wave.open(file_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # Sample rate
            wav_file.writeframes(audio_bytes)

        return response.json({"message": "Audio file uploaded successfully"}, status=200)
    else:
        return response.json({"message": "Missing file name or audio data"}, status=400)


async def process_youtube_download(url, job_uuid):
    class MyLogger(object):
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            print(msg)

    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    unique_filename = str(uuid.uuid4())
    save_path = f"/home/freiza/git_repos/TTS/TTS/server/ytdl/{unique_filename}.wav"

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
        'outtmpl': save_path,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Update the job status and completed_time in the database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("UPDATE jobs SET status = 'completed', completed_time = datetime('now') WHERE uuid = ?", (job_uuid,))
    conn.commit()
    conn.close()

@app.route("/api/youtube-dl")
async def youtubeClone(request):
    url = request.args.get("yturl", "")
    user = request.args.get("user", "")

    # Create a new job UUID
    job_uuid = str(uuid.uuid4())

    # Insert a new job into the jobs table
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO jobs (uuid, user, type, status, creation_time) VALUES (?, ?, ?, ?, datetime('now'))",
                   (job_uuid, user, "youtube-dl", "processing"))
    conn.commit()
    conn.close()

    # Process the YouTube download in the background
    asyncio.create_task(process_youtube_download(url, job_uuid))

    return response.json({"message": "YouTube clone job created successfully", "job_uuid": job_uuid})

@app.route("/api/job-status")
async def job_status(request):
    job_uuid = request.args.get("job_uuid", "")

    if not job_uuid:
        return response.json({"message": "Missing job UUID"}, status=400)

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT status, creation_time, completed_time FROM jobs WHERE uuid = ?", (job_uuid,))
    result = cursor.fetchone()
    conn.close()

    if result:
        status, creation_time, completed_time = result
        response_data = {
            "job_uuid": job_uuid,
            "status": status,
            "creation_time": creation_time,
            "completed_time": completed_time
        }
        return response.json(response_data)
    else:
        return response.json({"message": "Job not found"}, status=404)
