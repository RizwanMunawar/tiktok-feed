import os
import asyncio
import csv
import time
from datetime import datetime, timezone
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

async def runscreenshot(playwright: Playwright, url, screenshotpath):
    chromium = playwright.chromium # or "firefox" or "webkit".
    browser = await chromium.launch()
    page = await browser.new_page()
    await page.goto(url)
    # Save the screenshot
    await page.screenshot(path=screenshotpath, quality = 20, type = 'jpeg')
    await browser.close()

async def user_videos():
    with open('subscriptions.csv') as f:
        cf = csv.DictReader(f, fieldnames=['username'])
        for row in cf:
            user = row['username']
            print(f'Running for user \'{user}\'')

            # Initialize the feed generator
            fg = FeedGenerator()
            fg.id('https://www.tiktok.com/@' + user)
            fg.title(user + ' TikTok')
            fg.link(href='http://tiktok.com', rel='alternate')
            fg.logo(ghRawURL + 'tiktok-rss.png')
            fg.subtitle('All the latest TikToks from ' + user)
            fg.link(href=ghRawURL + 'rss/' + user + '.xml', rel='self')
            fg.language('en')

            # Fetch and process the latest video
            updated = None
            async with TikTokApi() as api:
                await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False)
                ttuser = api.user(user)

                try:
                    async for video in ttuser.videos(count=1):  # Fetch only the latest video
                        fe = fg.add_entry()
                        video_url = f"https://tiktok.com/@{user}/video/{video.id}"
                        
                        # Generate title and content with a fallback
                        title = video.as_dict.get('desc', 'No Title')[:255]
                        description = title  # Default content is title unless more detailed content available
                        
                        # If available, add the thumbnail as part of the description
                        if 'cover' in video.as_dict['video']:
                            thumbnail_url = video.as_dict['video']['cover']
                            description = f'<img src="{thumbnail_url}" alt="Video Thumbnail" /> {description}'

                        # Populate the entry
                        fe.id(video_url)  # Unique video URL as GUID
                        fe.title(title)
                        fe.link(href=video_url)
                        fe.description(description)
                        
                        # Timestamp the item and update feed's last modified date
                        publish_time = datetime.fromtimestamp(video.as_dict['createTime'], timezone.utc)
                        fe.pubDate(publish_time)
                        fg.updated(publish_time)
                        updated = publish_time if not updated else max(updated, publish_time)

                    # Set feed-level lastBuildDate to latest video timestamp
                    fg.lastBuildDate(updated)
                    fg.rss_file(f'rss/{user}.xml', pretty=True)  # Write the RSS feed to a file

                except Exception as e:
                    print(f"Error processing user {user}: {e}")

if __name__ == "__main__":
    asyncio.run(user_videos())
