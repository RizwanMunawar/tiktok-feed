import os
import asyncio
import csv
import time
from datetime import datetime, timezone, timedelta
from feedgen.feed import FeedGenerator
#from tiktokapipy.api import TikTokAPI
from TikTokApi import TikTokApi
import config
from playwright.async_api import async_playwright, Playwright
from pathlib import Path
from urllib.parse import urlparse


# Edit config.py to change your URLs
ghRawURL = config.ghRawURL

api = TikTokApi()

ms_token = os.environ.get(
    "MS_TOKEN", None
)

async def user_videos():
    with open('subscriptions.csv') as f:
        cf = csv.DictReader(f, fieldnames=['username'])
        for row in cf:
            user = row['username']
            print(f'Running for user \'{user}\'')

            # Initialize the feed generator fresh each time to avoid old entries
            fg = FeedGenerator()
            fg.id(f'https://www.tiktok.com/@{user}')
            fg.title(f"{user}'s TikTok Feed")
            fg.link(href='http://tiktok.com', rel='alternate')
            fg.logo(f"{ghRawURL}tiktok-rss.png")
            fg.subtitle(f"Latest TikToks from {user}")
            fg.link(href=f"{ghRawURL}rss/{user}.xml", rel='self')
            fg.language('en')

            # Define today and yesterday for date checking
            today = datetime.now(timezone.utc).date()
            yesterday = today - timedelta(days=3)

            # Fetch the latest video and only update if it was posted today or yesterday
            async with TikTokApi() as api:
                await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False)
                ttuser = api.user(user)

                try:
                    async for video in ttuser.videos(count=1):  # Fetch only the latest video
                        # Get the video's publish date
                        publish_time = datetime.fromtimestamp(video.as_dict['createTime'], timezone.utc)

                        # Check if the video was posted today or yesterday
                        if publish_time.date() in (today, yesterday):
                            # Clear any previous entries
                            fg._FeedGenerator__entries = []

                            fe = fg.add_entry()
                            video_url = f"https://tiktok.com/@{user}/video/{video.id}"

                            # Title and description
                            title = video.as_dict.get('desc', 'No Title')[:255]
                            description = title
                            if 'cover' in video.as_dict['video']:
                                thumbnail_url = video.as_dict['video']['cover']
                                description = f'<img src="{thumbnail_url}" alt="Video Thumbnail" /> {description}'

                            # Populate entry
                            fe.id(video_url)  # GUID as video URL
                            fe.title(title)
                            fe.link(href=video_url)
                            fe.description(description)

                            # Set publish date and feed's last updated date
                            fe.pubDate(publish_time)
                            fg.updated(publish_time)
                            fg.lastBuildDate(publish_time)

                            # Write the video (if posted today or yesterday) to RSS file
                            fg.rss_file(f'rss/{user}.xml', pretty=True)
                        else:
                            print(f"No new video posted today or yesterday for user {user}.")

                except Exception as e:
                    print(f"Error processing user {user}: {e}")

if __name__ == "__main__":
    asyncio.run(user_videos())
